from sakura.nl2test.models.decomposition import (
    DecompositionMode,
    Scenario,
    GrammaticalBlockList,
)
from sakura.nl2test.preprocessing.nl_decomposer import NLDecomposer
from sakura.utils.pretty.prints import pretty_print


def test_nl_grammatical_decomposition_grammatical():
    nl_description = "Ensure pet is added to owner and ID is generated."
    nl_decomposer = NLDecomposer(mode=DecompositionMode.GRAMMATICAL)
    grammatical_blocks: GrammaticalBlockList = nl_decomposer.decompose(nl_description)
    pretty_print("Grammatical blocks", grammatical_blocks)
    assert len(grammatical_blocks.grammatical_blocks) == 2

    blocks = grammatical_blocks.grammatical_blocks
    assert len(blocks[0].subjects) == 1
    assert blocks[0].subjects[0].lower() == "pet"
    assert len(blocks[0].prep_phrases) == 1
    assert blocks[0].prep_phrases[0].object.lower() == "owner"

    assert len(blocks[1].direct_objs) == 0
    assert len(blocks[1].subjects) == 1
    assert blocks[1].subjects[0].lower() == "id"


def test_nl_grammatical_decomposition_high_abs():
    nl_description = """
    Create a test case that verifies the ability to add a new pet to an owner's collection and persist it to the database.
    The test should check that the pet is correctly associated with the owner, that the owner's pet count increases by one, and that the database assigns a unique identifier to the new pet.
    Use JUnit 5 for test annotations and structure, along with AssertJ for making assertions about the state of the objects and the database.
    The test should also ensure that the transactional behavior is correctly applied, allowing for a clean and consistent state after the test execution.
    """
    nl_decomposer = NLDecomposer(mode=DecompositionMode.GRAMMATICAL)
    grammatical_blocks: GrammaticalBlockList = nl_decomposer.decompose(nl_description)
    assert grammatical_blocks is not None
    pretty_print("Grammatical blocks", grammatical_blocks)


def test_nl_gherkin_decomposition_high_abs():
    nl_description = """
    Create a test case that verifies the ability to add a new pet to an owner's collection and persist it to the database.
    The test should check that the pet is correctly associated with the owner, that the owner's pet count increases by one, and that the database assigns a unique identifier to the new pet.
    Use JUnit 5 for test annotations and structure, along with AssertJ for making assertions about the state of the objects and the database.
    The test should also ensure that the transactional behavior is correctly applied, allowing for a clean and consistent state after the test execution.
    """
    nl_decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)
    scenario: Scenario = nl_decomposer.decompose(nl_description)
    assert scenario is not None
    pretty_print("Gherkin-style scenario", scenario)


def test_nl_gherkin_decomposition_basic():
    nl_description = "User logs in and sees the dashboard."
    nl_decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)
    scenario = nl_decomposer.decompose(nl_description)
    pretty_print("Gherkin scenario", scenario)
    assert isinstance(scenario, Scenario)
    assert isinstance(scenario.setup, list)
    assert isinstance(scenario.gherkin_groups, list)
    assert isinstance(scenario.teardown, list)
