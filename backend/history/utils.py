from langchain_core.chat_history import BaseChatMessageHistory

# 세션별 대화 기록을 메모리에 저장하는 딕셔너리
store = {}

class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)
        
    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)

def get_by_session_id(session_id):
    """세션 ID별로 InMemoryHistory 인스턴스를 반환 (없으면 새로 생성)"""
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]