import re
from enum import Enum

from nltest.utils import Config
from nltest.utils.llm.call_llm import CallLLM
from nltest.utils.pretty.prints import pretty_print

class ClientType(Enum):
    CODE_GEN = "code_gen"
    SUMMARIZATION = "summarization"
    DECISION = "decision"
    STRUCTURED = "structured"

class LLMClient:
    def __init__(self, client_type: ClientType):

        # Parse the config
        config = Config()
        self.llm_provider = config.get("llm_provider", "name")
        self.model_id = config.get(self.llm_provider, "llm_model")
        self.api_url = config.get(self.llm_provider, "llm_api_url")

        if isinstance(client_type, ClientType):
            temp_key = f"{client_type.value}_temp"
        else:
            raise RuntimeError(f"Unknown client type: {client_type}")

        self.temp = config.get(self.llm_provider, temp_key)
        self.output_tokens = config.get(self.llm_provider, "output_tokens")

        self.llm = CallLLM()

    @staticmethod
    def sanitize(text: str) -> str:
        # Apply standard sanitation operations
        sanitize_operations = [
            lambda t: re.sub(r'(?si).*?</think>', '', t, flags=re.IGNORECASE), # Remove everything before and include </think>
        ]
        for op in sanitize_operations:
            text = op(text)
        return text.strip()

    def generate(self, prompt: str, *, temp: float = -1, sanitize: bool = False) -> str:
        response = self.llm.generate(
            input=prompt,
            provider=self.llm_provider,
            model_id=self.model_id,
            api_url=self.api_url,
            temp=self.temp if temp == -1 else temp,
            output_tokens=self.output_tokens,
        )
        if sanitize:
            return self.sanitize(response)
        return response
