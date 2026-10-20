"""Pinned neural embedding pipeline compatible with the vectors in qa.sql."""
import csv
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from build_index import dataset_digest

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_REVISION = '2c4055b12046f11709e9df2c122e59ffbdc2f900'
MAX_QUESTION_LENGTH = 2000
logger = logging.getLogger(__name__)


def load_categories(root=ROOT):
    with (root / 'qa_categories.json').open(encoding='utf-8') as handle:
        definitions = json.load(handle)['categories']
    with (root / 'QA.csv').open(encoding='utf-8-sig', newline='') as handle:
        questions = [row['question'] for row in csv.DictReader(handle)]
    categories = {}
    assigned = []
    for definition in definitions:
        name, entries = definition['name'], definition['questions']
        if not isinstance(name, str) or not name.strip() or name in categories:
            raise ValueError('Invalid or duplicate category name.')
        if not isinstance(entries, list) or not entries or any(not isinstance(q, str) or not q for q in entries):
            raise ValueError('Category questions must be nonempty strings.')
        categories[name] = tuple(entries)
        assigned.extend(entries)
    if not categories or len(assigned) != len(set(assigned)) or set(assigned) != set(questions):
        raise ValueError('Every QA question must be assigned to exactly one category.')
    return categories


class ServiceUnavailable(RuntimeError):
    """Safe, user-facing service error, without credentials or internal details."""


@dataclass(frozen=True)
class Match:
    question: str
    answer: str
    similarity: float


class Embedder:
    def __init__(self):
        directory = os.getenv('QA_MODEL_DIR')
        if directory:
            model = Path(directory) / 'model_quantized.onnx'
            tokenizer_path = Path(directory) / 'tokenizer.json'
        else:
            tokenizer_path = hf_hub_download(MODEL_ID, 'tokenizer.json', revision=MODEL_REVISION)
            model = hf_hub_download(MODEL_ID, 'onnx/model_quantized.onnx', revision=MODEL_REVISION)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.cls, self.sep, self.pad = [self.tokenizer.token_to_id(t) for t in ('<s>', '</s>', '<pad>')]
        if any(t is None for t in (self.cls, self.sep, self.pad)):
            raise ValueError('Unexpected model tokenizer.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.lock = Lock()

    def encode(self, text):
        # Keep the chunk size, special tokens and pooling identical to qa.sql.
        with self.lock:
            tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
            if not tokens:
                raise ValueError('Cannot embed empty text.')
            chunks = [tokens[i:i + 126] for i in range(0, len(tokens), 126)]
            width = max(map(len, chunks)) + 2
            ids = np.full((len(chunks), width), self.pad, dtype=np.int64)
            mask = np.zeros_like(ids)
            for i, chunk in enumerate(chunks):
                sequence = [self.cls] + chunk + [self.sep]
                ids[i, :len(sequence)] = sequence
                mask[i, :len(sequence)] = 1
            feed = dict(input_ids=ids, attention_mask=mask, token_type_ids=np.zeros_like(ids))
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]
        if hidden.shape != (*ids.shape, 384):
            raise ValueError('Unexpected embedding output dimensions.')
        pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1)[:, None]
        vector = np.average(pooled, axis=0, weights=[len(c) for c in chunks])
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm == 0:
            raise ValueError('Invalid embedding output.')
        return vector / norm


class LocalStore:
    def __init__(self, root=ROOT):
        with (root / 'QA.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['question', 'answer']:
                raise ValueError('Unexpected QA.csv columns.')
            self.rows = list(reader)
        if not self.rows or any(not r.get('question') or not r.get('answer') or None in r for r in self.rows):
            raise ValueError('Empty or malformed QA dataset.')
        with np.load(root / 'qa_index.npz', allow_pickle=False) as index:
            self.vectors = index['embeddings'].copy()
            digest = str(index['dataset_digest'].item())
        if digest != dataset_digest(self.rows):
            raise ValueError('QA.csv changed. Regenerate qa.sql and run build_index.py.')
        if self.vectors.shape != (len(self.rows), 384) or not np.isfinite(self.vectors).all():
            raise ValueError('Invalid QA index.')
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError('QA vectors are not normalized.')

    def search(self, vector, count=3, questions=None):
        allowed = None if questions is None else frozenset(questions)
        candidates = np.asarray([i for i, row in enumerate(self.rows) if allowed is None or row['question'] in allowed], dtype=np.int64)
        if not len(candidates):
            return []
        scores = np.clip(self.vectors[candidates] @ vector, -1, 1)
        ranked = np.argsort(-scores, kind='stable')[:count]
        return [Match(**self.rows[candidates[i]], similarity=float(scores[i])) for i in ranked]

    def exact(self, question, questions=None):
        if questions is not None and question not in questions:
            return None
        return next((Match(**r, similarity=1.0) for r in self.rows if r['question'].strip() == question), None)


class SupabaseStore:
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL', '').rstrip('/')
        self.key = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')
        parsed = urlparse(self.url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('SUPABASE_URL must be a valid HTTPS project URL.')
        if not self.key:
            raise ValueError('SUPABASE_SERVICE_ROLE_KEY is required.')

    def exact(self, question, questions=None):
        # Retrieval is delegated to the existing match_qa RPC in this mode.
        return None

    def search(self, vector, count=3, questions=None):
        payload = dict(query_embedding=vector.tolist(), match_count=count, match_threshold=-1.0)
        endpoint = 'match_qa'
        if questions is not None:
            if not questions:
                return []
            payload['category_questions'] = list(questions)
            endpoint = 'match_qa_in_category'
        body = json.dumps(payload).encode()
        request = Request(self.url + '/rest/v1/rpc/' + endpoint, data=body, method='POST', headers={
            'Content-Type': 'application/json', 'apikey': self.key, 'Authorization': 'Bearer ' + self.key,
        })
        try:
            with urlopen(request, timeout=20) as response:
                data = json.load(response)
            if not isinstance(data, list):
                raise ValueError('Invalid RPC response.')
            matches = [Match(question=r['question'], answer=r['answer'], similarity=float(r['similarity'])) for r in data]
            if any(not isinstance(m.question, str) or not isinstance(m.answer, str) or not np.isfinite(m.similarity) for m in matches):
                raise ValueError('Invalid RPC match.')
            if questions is not None:
                matches = [m for m in matches if m.question in questions]
            return sorted(matches, key=lambda m: m.similarity, reverse=True)
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            # Do not display or log response bodies, URLs, headers or service keys.
            logger.warning('Supabase retrieval failed (%s).', type(exc).__name__)
            raise ServiceUnavailable('目前無法查詢知識庫，請稍後再試或聯絡系統管理員。') from None


class QAService:
    def __init__(self, embedder=None, store=None, threshold=None):
        self.threshold = float(os.getenv('QA_MATCH_THRESHOLD', '0.45')) if threshold is None else threshold
        if not 0 <= self.threshold <= 1:
            raise ValueError('QA_MATCH_THRESHOLD must be between 0 and 1.')
        backend = os.getenv('QA_BACKEND', 'local').lower()
        if backend not in ('local', 'supabase'):
            raise ValueError('QA_BACKEND must be local or supabase.')
        self.store = store if store is not None else (LocalStore() if backend == 'local' else SupabaseStore())
        self.categories = load_categories()
        self.embedder = embedder if embedder is not None else Embedder()

    def answer(self, question, category=None):
        question = question.strip()
        if not question:
            return '請輸入您想詢問的問題。', ''
        if len(question) > MAX_QUESTION_LENGTH:
            return f'問題請控制在 {MAX_QUESTION_LENGTH} 字以內，並描述主要狀況或錯誤訊息。', ''
        if category is not None and category not in self.categories:
            return '請先選擇有效的問題分類，再提出問題。', ''
        allowed = self.categories[category] if category is not None else None
        exact = self.store.exact(question, questions=allowed)
        matches = [exact] if exact else self.store.search(self.embedder.encode(question), questions=allowed)
        if allowed is not None:
            matches = [m for m in matches if m.question in allowed]
        if not matches or matches[0].similarity < self.threshold:
            scope = f'「{category}」分類' if category else '知識庫'
            return f'目前{scope}沒有找到足夠相關的答案。請補充問題描述、切換適合的分類，或聯絡人工客服。', ''
        references = '\n\n'.join(f'{i}. {match.question}' for i, match in enumerate(matches, 1) if match.similarity >= self.threshold)
        return matches[0].answer, references
