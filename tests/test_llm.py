from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from nltest.test2nl.model.models import AbstractionLevel, TestDescriptionInfo
from nltest.utils.llm import LLMClient
from nltest.utils.llm.model import ClientType
from nltest.utils.pretty.prints import pretty_print


def test_llm_client(nl2test_context):
    llm = LLMClient(ClientType.SUMMARIZATION)

    system = "You are a helpful assistant that provides eloquent summaries with a British accent."
    query = "Why do people attend concerts? What is the purpose if they can just listen to the music at home?"

    messages = [SystemMessage(content=system), HumanMessage(content=query)]

    result = llm.invoke_messages(messages)
    assert result is not None
    assert isinstance(result, AIMessage)
    pretty_print("LLM Output", result)


def test_high_desc_prompt(test2nl_context):
    qualified_class_name = (
        "org.springframework.samples.petclinic.service.ClinicServiceTests"
    )
    method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

    _, prompt, is_successful = test2nl_context.test2nl_prompt.generate(
        method_signature, qualified_class_name, AbstractionLevel.HIGH
    )
    assert prompt, "Prompt was unsuccessfully rendered..."

    pretty_print("prompts", prompt)

    assert is_successful, "LLM generation was unsuccessful with prompts..."


def test_roundtrip_prompt(test2nl_context):
    qualified_class_name = (
        "org.springframework.samples.petclinic.service.ClinicServiceTests"
    )
    method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

    test_descriptions = test2nl_context.data_manager.load(
        "descriptions.json", TestDescriptionInfo
    )
    selected_description = test_descriptions[0]

    _, prompt, is_successful = test2nl_context.roundtrip_prompt.generate(
        method_signature, qualified_class_name, selected_description.description
    )
    assert prompt, "Prompt was unsuccessfully rendered..."

    pretty_print("prompts", prompt)

    assert is_successful, "LLM generation was unsuccessful with prompts..."
