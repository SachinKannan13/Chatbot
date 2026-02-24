from __future__ import annotations

import functools

from supabase import Client, create_client

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatHistoryStore:
    """Persist chatbot question/answer turns to Supabase."""

    def __init__(self):
        settings = get_settings()
        self._client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )

    def save_turn(self, question: str, answer: str, category: str = "") -> None:
        """
        Save a chat turn to Employees table.
        Uses existing columns: questions, answers.
        """
        # Try common table/column casings to handle quoted identifiers safely.
        table_candidates = ["Employees", "employees"]
        payload_candidates = [
            {"questions": question, "answers": answer, "category": category},
            {"Questions": question, "Answers": answer, "Category": category},
            {"questions": question, "answers": answer},
            {"Questions": question, "Answers": answer},
        ]

        last_error = ""
        for table_name in table_candidates:
            for payload in payload_candidates:
                try:
                    self._client.table(table_name).insert(payload).execute()
                    logger.info(
                        "chat_history_persisted",
                        table=table_name,
                        columns=list(payload.keys()),
                    )
                    return
                except Exception as e:
                    last_error = str(e)

        # Never break chat flow due to persistence issues.
        logger.warning("chat_history_persist_failed", error=last_error)


@functools.cache
def get_chat_history_store() -> ChatHistoryStore:
    return ChatHistoryStore()
