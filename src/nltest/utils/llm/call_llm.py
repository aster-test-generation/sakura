from typing import Any

import ollama
import replicate
import openai
from genai.schema import TextGenerationParameters
from openai import OpenAI, AzureOpenAI
import requests
from genai import Credentials, Client

from nltest.utils import constants
from nltest.utils.llm.providers import ProviderType, DecodingType


class CallLLM:

    def generate(
            self,
            input: str,
            provider: ProviderType,
            model_id: str,
            decode_type: DecodingType = DecodingType.sample,
            api_key: str = '',
            api_url: str = '',
            temp: float = 0.2,
            min_new_tokens: int = 100,
            output_tokens: int = 1024,
            stream: bool = False,
            k: int = 50,
            p: int = 1,
            stop_seq: list = ["END"]
    ) -> str:
        """
        Call LLM and generation the output
        Args:
            input: input prompts
            provider: LLM providers
            model_id: model name
            decode_type: decode mode, greedy or sample
            api_key: API key for the provider (not needed for Vela)
            api_url: API URL for the provider (not needed for OpenAI and Replicate)
            temp: A value used to modify the next-token probabilities in sampling mode.
            min_new_tokens: Minimum number of new tokens
            output_tokens: Maximum output tokens
            stream: Enables to stream partial progress as server-sent events. Defaults to false.
            k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            p: Similar to top_k except the candidates to generation the next token are
               the most likely tokens with probabilities that add up to at least top_p
            stop_seq: Stop sequences are one or more strings which will cause the text generation to
               stop if/when they are produced as part of the output.

        Returns:
        LLM generated text
        """
        match provider:

            case ProviderType.bam:
                llm_params = TextGenerationParameters(
                    min_new_tokens=min_new_tokens,
                    stop_sequences=stop_seq,
                    decoding_method=decode_type,
                    top_k=k,
                    top_p=p,
                    temperature=temp,
                    max_new_tokens=output_tokens,
                )
                auth_api_key = api_key
                auth_api_url = api_url
                creds = Credentials(auth_api_key, api_endpoint=auth_api_url)
                code_translator = Client(credentials=creds)
                responses = list(
                    code_translator.text.generation.create(
                        model_id=model_id,
                        inputs=[input],
                        parameters=llm_params
                    )
                )
                generated_text = ''
                for response in responses:
                    generated_text = response.results[0].generated_text
                    generated_text = generated_text.replace(input, "")
                return generated_text

            case ProviderType.rits:
                auth_api_key = api_key
                auth_api_url = api_url
                client = OpenAI(
                    api_key=api_key,
                    base_url=auth_api_url + model_id.lower().split('/')[-1].replace('.', '-') + "/v1",
                    default_headers={'RITS_API_KEY': auth_api_key}
                )

                generated_text = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": input,
                        }
                    ],
                    temperature=temp,
                    max_tokens=output_tokens,
                    frequency_penalty=0.0
                ).choices[0].message.content
                generated_text = generated_text.replace(input, "")
                return generated_text

            case ProviderType.vela:
                auth_api_url = api_url
                headers = {
                    'Content-Type': 'application/json',
                }
                response = requests.post(f'{auth_api_url}',
                                         headers=headers,
                                         json={'prompt': input,
                                               'model': model_id,
                                               "max_tokens": output_tokens,
                                               'temperature': temp
                                               })

                response.raise_for_status()
                result = response.json()
                generated_text = result.get('choices', '')[0]['text']
                return generated_text

            case ProviderType.azure:
                client = AzureOpenAI(
                    azure_endpoint=api_url.replace('MODEL_ID', model_id).replace('API_VERSION',
                                                                                 constants.AZURE_API_VERSION),
                    api_key=api_key,
                    api_version=constants.AZURE_API_VERSION
                )
                generated_text = client.chat.completions.create(
                    model=model_id,  # replace with the model of choice
                    messages=[
                        {"role": "user", "content": input},
                    ],
                    max_completion_tokens=output_tokens,
                    frequency_penalty=0.0
                ).choices[0].message.content
                generated_text = generated_text.replace(input, "")
                return generated_text

            case ProviderType.openai:
                client = openai.OpenAI(api_key=api_key)
                generated_text = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": input,
                        }
                    ],
                    temperature=temp,
                    max_tokens=output_tokens,
                    frequency_penalty=0.0
                ).choices[0].message.content
                generated_text = generated_text.replace(input, "")
                return generated_text

            case ProviderType.replicate:
                auth_api_key = api_key
                generated_text = replicate.run(
                    model_id,
                    input={
                        "prompts": input,
                        "max_tokens": output_tokens,
                        "min_tokens": min_new_tokens,
                        "temperature": temp,
                        "system_prompt": "",
                        "presence_penalty": 0,
                        "frequency_penalty": 0
                    })
                generated_text = generated_text.replace(input, "")
                return generated_text

            case ProviderType.ollama:
                generated_text = ollama.generate(model=model_id, prompt=input, options={"temperature": temp,
                                                                                        "num_predict": output_tokens,
                                                                                        "num_keep": min_new_tokens,
                                                                                        "presence_penalty": 0,
                                                                                        "frequency_penalty": 0
                                                                                        })
                generated_text = generated_text["response"]
                generated_text = generated_text.replace(input, "")
                return generated_text

            case _:
                raise NotImplementedError
