# conversation_state.py

conversation_state = {
    "last_intent": None,
    "last_subject": None,
    "history": []
}


def set_intent(intent: str):
    conversation_state["last_intent"] = intent


def get_intent() -> str | None:
    return conversation_state.get("last_intent")


def set_subject(subject: str):
    conversation_state["last_subject"] = subject


def get_subject():
    return conversation_state.get("last_subject")


def add_history(question: str, answer: str):
    conversation_state["history"].append((question, answer))


def get_last_answer() -> str | None:
    if conversation_state["history"]:
        return conversation_state["history"][-1][1]
    return None


def get_last_question() -> str | None:
    if conversation_state["history"]:
        return conversation_state["history"][-1][0]
    return None