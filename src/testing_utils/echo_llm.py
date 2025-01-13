from typing import Any, List, Optional
import time
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
)
from langchain_core.outputs import (
    ChatResult,
    ChatGeneration,
)


class EchoLLM(BaseChatModel):
    """Test LLM class that returns input as output

    delay (s): seconds to wait before answer.
    """
    delay: int = 0

    @property
    def _llm_type(self) -> str:
        """Return identifier of LLM type"""
        return "echo"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return last message content as output

        Note:
            Method is required by langchain interface.
            Do not use directly.
        """
        if not messages:
            return ChatResult(generations=[])

        message = AIMessage(content=messages[-1].content)
        generation = ChatGeneration(message=message)
        if self.delay > 0:
            time.sleep(self.delay)

        return ChatResult(generations=[generation])

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Combine multiple outputs into one

        Note:
            Method is required by langchain interface.
            Do not use directly.
        """
        return {}


def test_echo_llm():
    """Test that EchoLLM works as expected"""
    test_message = "test message"
    llm = EchoLLM()
    messages = [HumanMessage(content=test_message)]
    result = llm.invoke(messages)
    assert isinstance(result, AIMessage)
    assert result.content == test_message
