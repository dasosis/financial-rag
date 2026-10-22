"""
followup.py — Turn a follow-up ("and in 2023?") into a question that stands on its own.

Every question used to be answered in isolation, so "And in 2023?" after a
question about Microsoft's net income searched for "2023" and nothing else.
standalone_question() asks gpt-4o-mini to rewrite the latest question using the
last few turns of the conversation, resolving only what it refers back to. The
rewritten question then goes through the unchanged pipeline (planner, filing
choice, calculation or retrieval, answer check), and the API returns it so the
UI can show how the question was read.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from financial_rag import config

MAX_TURNS = 6           # most recent messages used (3 questions and their answers)
MAX_CHARS = 1500        # per message; long calculation answers are cut

REWRITE_PROMPT = """You rewrite the latest question in a conversation about company \
financial filings so that it can be understood without the conversation.

- Resolve what it refers back to: pronouns, "that", "the same", "and in 2023?", \
"what about Acme?", "how did it change?".
- Carry over the company, metric, fiscal years or topic from earlier turns only when the \
latest question relies on them. A question that names a different company or metric \
replaces the earlier one.
- If the latest question already stands on its own, return it unchanged.
- Never answer the question, and never add anything the user didn't ask for."""


class _Rewritten(BaseModel):
    question: str = Field(description="The latest question, rewritten to stand on its own")


def _transcript(history: list[tuple[str, str]]) -> str:
    lines = []
    for role, text in history[-MAX_TURNS:]:
        text = " ".join(text.split())
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + " …"
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


def standalone_question(question: str, history: list[tuple[str, str]]) -> str:
    """
    `question` rewritten to stand on its own, given earlier (role, text) turns,
    oldest first. Without history, or if the call fails, the question is
    returned as asked.
    """
    if not history:
        return question
    llm = config.chat_model(temperature=0, timeout=20, max_retries=1)
    try:
        result = llm.with_structured_output(_Rewritten).invoke([
            ("system", REWRITE_PROMPT),
            ("human", f"Conversation so far:\n{_transcript(history)}\n\n"
                      f"Latest question: {question}"),
        ])
    except config.GatewayLimit:
        raise   # a spent budget is the user's answer, not a reason to degrade it
    except Exception as exc:
        print(f"Couldn't rewrite the follow-up, answering it as asked: {exc}")
        return question
    return result.question.strip() or question
