"""Streamlit customer-support app; deploy with app.py as the entry point."""
import logging
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from qa_service import MAX_QUESTION_LENGTH, QAService, ServiceUnavailable, load_categories
except ModuleNotFoundError as exc:
    # Show fixed deployment instructions, without exposing tracebacks or secrets.
    missing = (exc.name or '').split('.')[0]
    if missing in ('qa_service', 'build_index'):
        st.error(f'缺少程式檔案：{missing}.py。請上傳至 GitHub，與 app.py 放在同一個目錄，並確認檔名大小寫相同。')
    else:
        packages = {'huggingface_hub': 'huggingface-hub', 'numpy': 'numpy', 'onnxruntime': 'onnxruntime', 'tokenizers': 'tokenizers'}
        package = packages.get(missing)
        detail = f'缺少 Python 套件：{package}。' if package else 'Python 相依套件不完整。'
        st.error(detail + '請上傳最新版 requirements.txt，與 app.py 放在同一個目錄，並重新部署。')
        st.info('部署目錄請使用部署包內容，避免舊 uv.lock 優先於 requirements.txt 被採用。')
    st.stop()

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_service(category_definitions):
    # Include the category mapping in the cache key so reassignment reloads the service.
    try:
        settings = st.secrets.to_dict()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        settings = {}
    for name in ('QA_BACKEND', 'QA_MATCH_THRESHOLD', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'QA_MODEL_DIR'):
        if name in settings:
            os.environ.setdefault(name, str(settings[name]))
    return QAService()


def respond(message, category=None):
    if category is None:
        return '請先從左側選擇問題分類，再提出問題。', ''
    if not isinstance(message, str) or not message.strip():
        return '請輸入您想詢問的問題。', ''
    if len(message.strip()) > MAX_QUESTION_LENGTH:
        return f'問題請控制在 {MAX_QUESTION_LENGTH} 字以內，並描述主要狀況或錯誤訊息。', ''
    try:
        return get_service(load_categories()).answer(message, category=category)
    except ServiceUnavailable as exc:
        return str(exc), ''
    except Exception as exc:
        logger.error('QA request failed (%s).', type(exc).__name__)
        return '客服系統暫時無法回覆，請稍後重試。若持續發生，請通知系統管理員檢查模型與知識庫設定。', ''


def show_message(message):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('references'):
            with st.expander('參考問答'):
                st.markdown(message['references'])


def main():
    st.set_page_config(page_title='FCCL 銅箔檢測｜線上客服', page_icon='🔎', layout='centered')
    st.title('FCCL 銅箔檢測・線上客服')
    st.caption('先選擇問題分類，再描述您的狀況，查詢該分類的客服知識庫。')
    st.html('''<style>
    [data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {
        min-height: calc(100dvh - 7rem);
    }
    .st-key-sidebar_footer { margin-top: auto; padding-top: 1rem; }
    </style>''')
    try:
        categories = load_categories()
    except (OSError, ValueError, KeyError, TypeError):
        st.error('無法讀取問題分類。請確認最新版 qa_categories.json 與 QA.csv 已上傳至 app.py 的同一目錄。')
        st.stop()
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    selected_question = None
    with st.sidebar:
        st.header('客服小幫手')
        category = st.selectbox('問題分類', list(categories), index=None, placeholder='請先選擇問題分類', key='category')
        st.session_state.messages = st.session_state.conversations.setdefault(category, []) if category else []
        if category:
            st.caption(f'目前分類包含 {len(categories[category])} 組問答。')
            st.subheader('本分類常見問題')
            for i, question in enumerate(categories[category][:3]):
                if st.button(question, key=f'example_{category}_{i}', width='stretch'):
                    selected_question = question
        else:
            st.caption('選擇分類後，即可查看常見問題並開始提問。')
        st.divider()
        st.caption('每次僅查詢所選分類。切換分類時，會顯示該分類的對話紀錄。')
        st.caption('回答依據既有客服資料。設備參數、功能與品質判定，請以現場核准文件為準。')
        with st.container(key='sidebar_footer'):
            if st.button('清除對話', key='clear_chat', width='stretch', disabled=category is None, help='清除目前分類的對話紀錄'):
                st.session_state.conversations[category] = []
                st.session_state.messages = []
                st.rerun()
    if category:
        st.subheader(category)
    else:
        st.info('請先從左側選擇問題分類，選擇後即可開始提問。')
    if not st.session_state.messages:
        with st.chat_message('assistant'):
            st.write(f'您好！您可以詢問「{category}」相關問題，或從左側選擇本分類的常見問題。' if category else '您好！請先選擇您需要協助的問題分類。')
    for message in st.session_state.messages:
        show_message(message)
    typed_question = st.chat_input(f'請輸入「{category}」相關問題' if category else '請先選擇問題分類', max_chars=MAX_QUESTION_LENGTH, disabled=category is None)
    question = typed_question or selected_question
    if category and question:
        message = {'role': 'user', 'content': question}
        st.session_state.messages.append(message)
        show_message(message)
        with st.spinner('正在查詢知識庫，首次使用需載入模型…'):
            answer, references = respond(question, category=category)
        reply = {'role': 'assistant', 'content': answer, 'references': references}
        st.session_state.messages.append(reply)
        show_message(reply)


if __name__ == '__main__':
    main()
