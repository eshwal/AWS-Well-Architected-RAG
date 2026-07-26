from abc import ABC, abstractmethod
from typing import Any
from src.models.model import ChatResponse, RetrievedChunk
from src.service.rag import query_compliance_platform



class SkippedIntent(Exception):
    pass

class Invoker(ABC):

    @abstractmethod
    async def invoke(
        self, question: str, flags: dict, intent: str
    ) -> ChatResponse:
        ...


class ServiceInvoker(Invoker):
    SUPPORTED_INTENTS = {"rag"}

    async def invoke(
        self, question: str, flags: dict, intent: str
    ) -> ChatResponse:
        if intent not in self.SUPPORTED_INTENTS:
            raise SkippedIntent(f"intent={intent} not supported in service mode")

        return await query_compliance_platform(question, flags)
