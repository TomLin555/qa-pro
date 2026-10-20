"""Extract existing neural embeddings from qa.sql without recomputing them."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent


def dataset_digest(rows):
    payload = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def build_index(root=ROOT):
    with (root / 'QA.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['question', 'answer']:
            raise ValueError('QA.csv must contain only question and answer columns.')
        rows = list(reader)
    pattern = r"\('((?:[^']|'')*)', '((?:[^']|'')*)', '(\[[^']+\])'::vector\(384\)\)"
    matches = re.findall(pattern, (root / 'qa.sql').read_text(encoding='utf-8'))
    sql_rows = [dict(question=q.replace("''", "'"), answer=a.replace("''", "'")) for q, a, _ in matches]
    if not rows or sql_rows != rows:
        raise ValueError('QA.csv and qa.sql do not contain the same ordered QA records. Regenerate SQL embeddings first.')
    vectors = np.asarray([json.loads(v) for _, _, v in matches], dtype=np.float32)
    if vectors.shape != (len(rows), 384) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding dimensions or values.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Embeddings must be L2-normalized.')
    output = root / 'qa_index.npz'
    np.savez_compressed(output, embeddings=vectors, dataset_digest=dataset_digest(rows))
    return output, len(rows)


if __name__ == '__main__':
    path, count = build_index()
    print(f'Created {path.name}: {count} records, 384 dimensions.')
