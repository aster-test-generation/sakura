from typing import List
from unittest import TestCase
from pathlib import Path
from unittest.mock import MagicMock

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.nl2test.generation.localization import LocalizationOrchestrator, LocalizationTools
from nltest.nl2test.model.models import GrammaticalBlock, AtomicBlock
from nltest.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer, ProjectIndexer
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher, ProjectSearcher
from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore
from nltest.nl2test.preprocessing.embedders import HttpEmbedder, OllamaEmbedder
from nltest.nl2test.preprocessing.extractors import MethodSnippetExtractor
from nltest.utils import Config
from nltest.utils.llm_configs import LLM_CONFIG, EMB_CONFIG
from nltest.utils.pretty.prints import pretty_print


class TestNL2Test(TestCase):
    def setUp(self) -> None:

        # === User-Defined ===
        project_name = "spring-petclinic"

        # llm_model = "LLAMA-3.1-8B"
        # llm_model = "DEVSTRAL-24B"
        llm_model = "DEEPSEEK-R1"

        emb_model = "NOMIC-AI-EMB-7B"
        # emb_model = "QWEN-3-EMB-8B" # Embedding model

        summarization_temp = 0.4
        code_gen_temp = 0.5
        decision_temp = 0.3
        structured_temp = 0.3

        project_root = Path(f"./resources/{project_name}")
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(f"Project root directory {project_root} does not exist.")

        # Config default assignments - TODO: Regularize this
        config = Config(None, reuse=False)
        config.set("llm_provider", "name", "VELA")
        config.set("emb_provider", "name", "VELA")

        output_dir = Path(f"./output/{project_name}")

        # Assign VELA LLM model configs
        llm_provider = config.get("llm_provider", "name")
        config.set(llm_provider, "llm_model", val=LLM_CONFIG[llm_model]["identifier"])
        config.set(llm_provider, "llm_api_url", val=LLM_CONFIG[llm_model]["api_url"])
        config.set(llm_provider, "summarization_temp", summarization_temp)
        config.set(llm_provider, "code_gen_temp", code_gen_temp)
        config.set(llm_provider, "decision_temp", decision_temp)
        config.set(llm_provider, "structured_temp", structured_temp)
        config.set(llm_provider, "output_tokens",
                   val=LLM_CONFIG[llm_model]["output_tokens"] if "output_tokens" in LLM_CONFIG[llm_model] else 10000)

        # Assign VELA VectorStore model configs
        emb_provider = config.get("emb_provider", "name")
        config.set(emb_provider, "emb_model", val=EMB_CONFIG[emb_model]["identifier"])
        config.set(emb_provider, "emb_api_url", val=EMB_CONFIG[emb_model]["api_url"])

        self.config = Config()

        # Generate analysis of the current project
        self.analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=False,
        )

    def test_method_vector_store_single_method_ollama(self):
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "getPet(Integer)"

        method_details = self.analysis.get_method(qualified_class_name, method_signature)
        self.assertIsNotNone(method_details)

        provider = self.config.get("emb_provider", "name").lower()
        emb_model = self.config.get(provider, "emb_model")

        embedder = OllamaEmbedder(
            model_id=emb_model,
        )
        vector_store = MethodVectorStore(embedder)
        method_snippet = MethodSnippetExtractor(self.analysis).get_method_snippet(qualified_class_name,
                                                                                  method_signature)
        vector_store.add_snippets([method_snippet])

        searcher = MethodSearcher(vector_store)

        desc_str = """Ensure that the system correctly inserts a new pet into the database and generates an identifier for it. 
                The test should validate that when a pet is added to an owner's collection, the database reflects this addition 
                with an incremented count of pets for that owner. Additionally, it should confirm that the newly created pet is 
                assigned a non-null ID after being persisted to the database. This test employs JUnit 5 for test structure and 
                AssertJ for making assertions, ensuring that the database operations and entity state transitions behave as expected.
                """

        search_results = searcher.find_similar(desc_str)
        pretty_print("Search results", search_results)
        return searcher

    def test_method_vector_store_single_method_http(self):
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "getPet(java.lang.Integer)"

        method_details = self.analysis.get_method(qualified_class_name, method_signature)
        self.assertIsNotNone(method_details)

        provider = self.config.get("emb_provider", "name")
        emb_model = self.config.get(provider, "emb_model")
        api_url = self.config.get(provider, "emb_api_url")

        embedder = HttpEmbedder(
            model_id=emb_model,
            api_url=api_url,
        )
        vector_store = MethodVectorStore(embedder)
        method_snippet = MethodSnippetExtractor(self.analysis).get_method_snippet(qualified_class_name,
                                                                                  method_signature)
        vector_store.add_snippets([method_snippet])

        searcher = MethodSearcher(vector_store)

        desc_str = """Ensure that the system correctly inserts a new pet into the database and generates an identifier for it. 
                The test should validate that when a pet is added to an owner's collection, the database reflects this addition 
                with an incremented count of pets for that owner. Additionally, it should confirm that the newly created pet is 
                assigned a non-null ID after being persisted to the database. This test employs JUnit 5 for test structure and 
                AssertJ for making assertions, ensuring that the database operations and entity state transitions behave as expected.
                """

        num_similar = 1
        search_results = searcher.find_similar(desc_str, k=num_similar)
        pretty_print("Search results", search_results)
        self.assertEqual(len(search_results), num_similar)

    def test_method_vector_store(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        method_searcher = MethodIndexer(self.analysis).build_index()

        desc_str = """Ensure that the system correctly inserts a new pet into the database and generates an identifier for it. 
        The test should validate that when a pet is added to an owner's collection, the database reflects this addition 
        with an incremented count of pets for that owner. Additionally, it should confirm that the newly created pet is 
        assigned a non-null ID after being persisted to the database. This test employs JUnit 5 for test structure and 
        AssertJ for making assertions, ensuring that the database operations and entity state transitions behave as expected.
        """

        num_similar = 5
        search_results = method_searcher.find_similar(desc_str, k=num_similar)
        pretty_print("Most similar to complete description", search_results)
        self.assertEqual(len(search_results), num_similar)

        num_similar = 3
        desc_substr = "Get the owner from repository with ID 6 and make sure it exists."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get the owner by ID", f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById")

        desc_substr = "Get the owner from the repository return."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get the owner from repository return", f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get")

        desc_substr = "Create a pet."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Create a pet by ID", f"FOUND: {search_results}\nACTUAL: default Pet constructor")

        desc_substr = "Add the pet to the owner."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Add pet to owner", f"FOUND: {search_results}\nACTUAL: Owner.addPet")

        desc_substr = "Save the owner to the repository."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Save the owner",
                     f"FOUND: {search_results}\nACTUAL: OwnerRepository.save (inherited from JpaRepository library class)")

        desc_substr = "Again, get the owner from the repository with ID 6"
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get the owner by ID", f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById")

        desc_substr = "Get the owner from the repository return."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get the owner from repository return", f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get")

        desc_substr = "Check that the total count of pets in the owner is correct."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get all pets and check size", f"FOUND: {search_results}\nACTUAL: Owner.getPets")

        desc_substr = "Get the added pet from the owner."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get pet in owner", f"FOUND: {search_results}\nACTUAL: Owner.getPet")

        desc_substr = "Get the pet id and check it is non-null."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get pet id", f"FOUND: {search_results}\nACTUAL: Pet.getId")

    def test_proj_vector_store_single_search(self):
        proj_searcher: ProjectSearcher = ProjectIndexer(self.analysis).build_index()
        class_searcher: ClassSearcher = ClassIndexer(self.analysis).build_index()

        num_similar = 5
        desc_substr = "Get the owner from repository with ID 6 and make sure it exists."

        method_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        class_results = proj_searcher.find_classes_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get the owner by ID", f"METHOD FOUND: {method_results}\nACTUAL: OwnerRepository.findById")
        pretty_print("Get the owner by ID", f"CLASS FOUND: {class_results}\nACTUAL: OwnerRepository")

        class_results = class_searcher.find_similar_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get the owner by ID", f"CLASS FOUND: {class_results}\nACTUAL: OwnerRepository")

    def test_proj_vector_store_complete(self):
        proj_searcher: ProjectSearcher = ProjectIndexer(self.analysis).build_index()

        num_similar = 3
        desc_substr = "Get the owner from repository with ID 6 and make sure it exists."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get the owner by ID", f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById")

        desc_substr = "Get the owner from the repository return."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get the owner from repository return", f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get")

        desc_substr = "Create a pet."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        search_results = proj_searcher.find_classes_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Create a pet by ID", f"FOUND: {search_results}\nACTUAL: default Pet constructor")

        desc_substr = "Add the pet to the owner."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Add pet to owner", f"FOUND: {search_results}\nACTUAL: Owner.addPet")

        desc_substr = "Save the owner to the repository."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        search_results = proj_searcher.find_classes_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Save the owner",
                     f"FOUND: {search_results}\nACTUAL: OwnerRepository.save (inherited from JpaRepository library class)")

        desc_substr = "Again, get the owner from the repository with ID 6"
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get the owner by ID", f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById")

        desc_substr = "Get the owner from the repository return."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get the owner from repository return", f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get")

        desc_substr = "Check that the total count of pets in the owner is correct."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get all pets and check size", f"FOUND: {search_results}\nACTUAL: Owner.getPets")

        desc_substr = "Get the added pet from the owner."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get pet in owner", f"FOUND: {search_results}\nACTUAL: Owner.getPet")

        desc_substr = "Get the pet id and check it is non-null."
        search_results = proj_searcher.find_methods_in_range(desc_substr, i=1, j=num_similar)
        pretty_print("Get pet id", f"FOUND: {search_results}\nACTUAL: Pet.getId")

    def test_nl_decomposition(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        nl_decomposer = NLDecomposer()
        grammatical_blocks: List[GrammaticalBlock] = nl_decomposer.decompose(nl_description)
        pretty_print("Grammatical blocks", grammatical_blocks)
        self.assertEqual(len(grammatical_blocks), 2)

        self.assertEqual(len(grammatical_blocks[0].direct_objs), 1)
        self.assertEqual(grammatical_blocks[0].direct_objs[0].lower(), "pet")
        self.assertEqual(len(grammatical_blocks[0].prep_phrases), 1)
        self.assertEqual(len(grammatical_blocks[0].prep_phrases[0]), 2)
        self.assertEqual(grammatical_blocks[0].prep_phrases[0][1].lower(), "owner")

        self.assertEqual(len(grammatical_blocks[1].direct_objs), 1)
        self.assertEqual(grammatical_blocks[1].direct_objs[0].lower(), "id")

    def test_nl_decomposition_high_abs(self):
        nl_description = """
        Create a test case that verifies the ability to add a new pet to an owner's collection and persist it to the database. 
        The test should check that the pet is correctly associated with the owner, that the owner's pet count increases by one, and that the database assigns a unique identifier to the new pet. 
        Use JUnit 5 for test annotations and structure, along with AssertJ for making assertions about the state of the objects and the database. 
        The test should also ensure that the transactional behavior is correctly applied, allowing for a clean and consistent state after the test execution.
        """
        nl_decomposer = NLDecomposer()
        grammatical_blocks: List[GrammaticalBlock] = nl_decomposer.decompose(nl_description)
        pretty_print("Grammatical blocks", grammatical_blocks)

    def test_localization_call_site_tool(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        fake_method_searcher = MagicMock()
        fake_class_searcher = MagicMock()
        fake_llm = MagicMock()

        localization_tools = LocalizationTools(analysis=self.analysis, method_searcher=fake_method_searcher,
                                               class_searcher=fake_class_searcher, structured_llm=fake_llm)

        call_site_tool = localization_tools._make_call_site_details_tool()

        cleaned_call_sites = call_site_tool.func(qualified_class_name, method_signature)
        pretty_print("Cleaned call site details", cleaned_call_sites)

    def test_localization_modify_atomic_block_tool(self):
        pass

    def test_localization_agent_simple(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        nl_decomposer = NLDecomposer()
        grammatical_blocks: List[GrammaticalBlock] = nl_decomposer.decompose(nl_description)

        method_searcher = MethodIndexer(self.analysis).build_index()
        class_searcher = ClassIndexer(self.analysis).build_index()

        atomic_blocks: List[AtomicBlock] = [
            AtomicBlock(**gb.model_dump())
            for gb in grammatical_blocks
        ]
        pretty_print("Initial atomic blocks", atomic_blocks)

        supervisor_instructions = "Find the relevant methods and refine the atomic blocks."

        localization_agent = LocalizationOrchestrator(self.analysis, method_searcher, class_searcher, nl_description)
        refined_blocks = localization_agent.assign_task(supervisor_instructions, atomic_blocks)
        pretty_print("Refined atomic blocks", refined_blocks)
