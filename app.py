"""Streamlit customer-support app; deploy with app.py as the entry point."""
import logging
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from qa_service import MAX_QUESTION_LENGTH, QAService, ServiceUnavailable

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_service():
    try:
        settings = st.secrets.to_dict()
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        settings = {}
    for name in ('QA_BACKEND', 'QA_MATCH_THRESHOLD', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'QA_MODEL_DIR'):
        if name in settings:
            os.environ.setdefault(name, str(settings[name]))
    return QAService()


def respond(message):
    if not isinstance(message, str) or not message.strip():
        return '請輸入您想詢問的問題。', ''
    if len(message.strip()) > MAX_QUESTION_LENGTH:
        return f'問題請控制在 {MAX_QUESTION_LENGTH} 字以內，並描述主要狀況或錯誤訊息。', ''
    try:
        return get_service().answer(message)
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
    st.caption('描述操作狀況、缺陷類型或錯誤訊息，即可查詢客服知識庫。')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    selected_question = None
    with st.sidebar:
        st.header('客服小幫手')
        st.markdown('請提供遇到的現象、缺陷名稱或錯誤訊息，讓問題更明確。每一則問題會獨立查詢。')
        st.subheader('常見問題')
        for question in ('忘記系統登入密碼怎麼辦？', '影像模糊會影響辨識嗎？', '如何匯出檢測報告？'):
            if st.button(question, width='stretch'):
                selected_question = question
        if st.button('清除對話', key='clear_chat', width='stretch'):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption('回答依據既有客服資料。設備參數、功能與品質判定，請以現場核准文件為準。')
    if not st.session_state.messages:
        with st.chat_message('assistant'):
            st.write('您好！請輸入 FCCL 銅箔 AI 檢測系統的相關問題，或從左側選擇常見問題。')
    for message in st.session_state.messages:
        show_message(message)
    typed_question = st.chat_input('例如：相機連線中斷，該如何排除？', max_chars=MAX_QUESTION_LENGTH)
    question = typed_question or selected_question
    if question:
        message = {'role': 'user', 'content': question}
        st.session_state.messages.append(message)
        show_message(message)
        with st.spinner('正在查詢知識庫，首次使用需載入模型…'):
            answer, references = respond(question)
        reply = {'role': 'assistant', 'content': answer, 'references': references}
        st.session_state.messages.append(reply)
        show_message(reply)


if __name__ == '__main__':
    main()
