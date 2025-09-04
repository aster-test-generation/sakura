from nltest.nl2test.models.decomposition import (
    DecompositionMode,
    Scenario,
    GrammaticalBlockList,
)
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.utils.pretty.prints import pretty_print

from tests._base_nl2test import BaseNL2Test


class TestNLDecomposition(BaseNL2Test):
    def test_nl_grammatical_decomposition_grammatical(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        nl_decomposer = NLDecomposer(mode=DecompositionMode.GRAMMATICAL)
        grammatical_blocks: GrammaticalBlockList = nl_decomposer.decompose(
            nl_description
        )
        pretty_print("Grammatical blocks", grammatical_blocks)
        self.assertEqual(len(grammatical_blocks.grammatical_blocks), 2)

        blocks = grammatical_blocks.grammatical_blocks
        self.assertEqual(len(blocks[0].subjects), 1)
        self.assertEqual(blocks[0].subjects[0].lower(), "pet")
        self.assertEqual(len(blocks[0].prep_phrases), 1)
        self.assertEqual(blocks[0].prep_phrases[0].object.lower(), "owner")

        self.assertEqual(len(blocks[1].direct_objs), 0)
        self.assertEqual(len(blocks[1].subjects), 1)
        self.assertEqual(blocks[1].subjects[0].lower(), "id")

    def test_nl_grammatical_decomposition_high_abs(self):
        nl_description = """
        Create a test case that verifies the ability to add a new pet to an owner's collection and persist it to the database. 
        The test should check that the pet is correctly associated with the owner, that the owner's pet count increases by one, and that the database assigns a unique identifier to the new pet. 
        Use JUnit 5 for test annotations and structure, along with AssertJ for making assertions about the state of the objects and the database. 
        The test should also ensure that the transactional behavior is correctly applied, allowing for a clean and consistent state after the test execution.
        """
        nl_decomposer = NLDecomposer(mode=DecompositionMode.GRAMMATICAL)
        grammatical_blocks: GrammaticalBlockList = nl_decomposer.decompose(
            nl_description
        )
        self.assertIsNotNone(grammatical_blocks)
        pretty_print("Grammatical blocks", grammatical_blocks)

    def test_nl_gherkin_decomposition_high_abs(self):
        nl_description = """
        Create a test case that verifies the ability to add a new pet to an owner's collection and persist it to the database. 
        The test should check that the pet is correctly associated with the owner, that the owner's pet count increases by one, and that the database assigns a unique identifier to the new pet. 
        Use JUnit 5 for test annotations and structure, along with AssertJ for making assertions about the state of the objects and the database. 
        The test should also ensure that the transactional behavior is correctly applied, allowing for a clean and consistent state after the test execution.
        """
        nl_decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)
        scenario: Scenario = nl_decomposer.decompose(nl_description)
        self.assertIsNotNone(scenario)
        pretty_print("Gherkin-style scenario", scenario)

    def test_nl_gherkin_decomposition_basic(self):
        nl_description = "User logs in and sees the dashboard."
        nl_decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)
        scenario = nl_decomposer.decompose(nl_description)
        pretty_print("Gherkin scenario", scenario)
        # Basic structural assertions
        self.assertIsInstance(scenario, Scenario)
        self.assertIsInstance(scenario.testing_framework, str)
        self.assertIsInstance(scenario.setup, list)
        self.assertIsInstance(scenario.steps, list)
        self.assertIsInstance(scenario.teardown, list)
