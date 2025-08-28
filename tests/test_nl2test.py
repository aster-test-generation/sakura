import os

from dotenv import load_dotenv
from typing import List
from unittest import TestCase
from pathlib import Path
from unittest.mock import MagicMock

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from nltest.nl2test.generation.localization import LocalizationOrchestrator, LocalizationTools
from nltest.nl2test.models import GrammaticalBlock, AtomicBlock, NL2TestInput, CandidateMethod, AbstractionLevel
from nltest.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer, ProjectIndexer
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher, ProjectSearcher
from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore
from nltest.nl2test.preprocessing.embedders import HttpEmbedder, OllamaEmbedder
from nltest.nl2test.preprocessing.extractors import MethodSnippetExtractor
from nltest.utils.config import Config, init_config
from nltest.utils.llm import LLMClient, usage_tracker
from nltest.utils.llm.model import Provider, ClientType
from nltest.utils.pretty.prints import pretty_print
from nltest.nl2test.evaluation.localization_grader import LocalizationGrader
from nltest.nl2test.pipeline import Pipeline


class TestNL2Test(TestCase):
    def setUp(self) -> None:
        load_dotenv()

        # === User-Defined ===
        project_name = "spring-petclinic"

        llm_model = "deepseek/deepseek-chat-v3-0324"
        llm_model = "openai/gpt-5-mini"
        # llm_model = "qwen/qwen3-coder"
        # llm_model = "moonshotai/kimi-k2"
        # llm_model = "mistralai/devstral-small"
        llm_model = "mistralai/devstral-medium"
        emb_model = "nomic-embed-text:v1.5"

        project_root = Path(f"./resources/{project_name}")
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(f"Project root directory {project_root} does not exist.")
        output_dir = Path(f"./output/{project_name}")
        print(output_dir.resolve())

        self.config = init_config(
            project_name=project_name,
            base_project_dir="./resources",
            output_dir=str(output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            emb_provider=Provider.OLLAMA,
            emb_model=emb_model,
            llm_api_key=os.getenv("OPENROUTER_API_KEY"), # Assign from env
            emb_api_key=None,
        )

        # Generate analysis of the current project
        self.analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=False,
        )

        usage_tracker.reset()

    def test_llm_client(self):
        llm = LLMClient(ClientType.SUMMARIZATION)

        system = "You are a helpful assistant that provides eloquent summaries with a British accent."
        query = "Why do people attend concerts? What is the purpose if they can just listen to the music at home?"

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=query)
        ]

        result = llm.invoke_messages(messages)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AIMessage)
        pretty_print("LLM Output", result)

    def test_method_vector_store_single_method_ollama(self):
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "getPet(java.lang.Integer)"

        method_details = self.analysis.get_method(qualified_class_name, method_signature)
        self.assertIsNotNone(method_details)

        emb_model = self.config.get("emb", "model")

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

        self.assertEqual(len(grammatical_blocks[0].subjects), 1)
        self.assertEqual(grammatical_blocks[0].subjects[0].lower(), "pet")
        self.assertEqual(len(grammatical_blocks[0].prep_phrases), 1)
        self.assertEqual(grammatical_blocks[0].prep_phrases[0].object.lower(), "owner")

        self.assertEqual(len(grammatical_blocks[1].direct_objs), 0)
        self.assertEqual(len(grammatical_blocks[1].subjects), 1)
        self.assertEqual(grammatical_blocks[1].subjects[0].lower(), "id")

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
            AtomicBlock.from_grammatical_block(gb) for gb in grammatical_blocks
        ]
        pretty_print("Initial atomic blocks", atomic_blocks)

        supervisor_instructions = "Find the relevant methods and refine the atomic blocks."

        nl2_input = NL2TestInput(
            description=nl_description,
            project_name="spring-petclinic",
            qualified_class_name="",
            method_signature="",
        )

        localization_agent = LocalizationOrchestrator(self.analysis, method_searcher, class_searcher, nl2_input)

        usage_tracker.start()
        refined_blocks, comments = localization_agent.assign_task(supervisor_instructions, atomic_blocks)
        prices = usage_tracker.stop()

        pretty_print("Refined atomic blocks", refined_blocks)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_localization_grader(self):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.owner.OwnerControllerTests",
            method_signature="testProcessCreationFormSuccess()",
            description="Test that the owner creation form processes successfully when valid data is submitted.",
            project_name="spring-petclinic",
        )
        
        atomic_blocks = [
            AtomicBlock(
                order=0,
                subjects=["form"],
                verbs=["process"],
                past_participles=[],
                direct_objs=["creation"],
                indirect_objs=[],
                prep_phrases=[],
                polarity="positive",
                conditions=[],
                simplified="form processes creation",
                candidate_methods=[
                    CandidateMethod(
                        implementing_class_name="org.springframework.samples.petclinic.owner.OwnerController",
                        containing_class_name="org.springframework.samples.petclinic.owner.OwnerController",
                        method_signature="processCreationForm(Owner, BindingResult, ModelMap)"
                    )
                ],
                notes=""
            )
        ]
        
        grader = LocalizationGrader(nl2_input, self.analysis, self.config.get("project", "base_project_dir"))
        
        # Test basic grading
        coverage_score, detailed_results = grader.grade(atomic_blocks, detailed_output=False)
        self.assertIsInstance(coverage_score, float)
        self.assertGreaterEqual(coverage_score, 0.0)
        self.assertLessEqual(coverage_score, 1.0)
        self.assertIsNone(detailed_results)
        
        # Test detailed grading
        coverage_score, detailed_results = grader.grade(atomic_blocks, detailed_output=True)
        self.assertIsInstance(coverage_score, float)
        self.assertGreaterEqual(coverage_score, 0.0)
        self.assertLessEqual(coverage_score, 1.0)
        self.assertIsNotNone(detailed_results)
        self.assertIsInstance(detailed_results, dict)
        
        expected_keys = [
            "test_class", "test_method", "total_focal_methods", "covered_focal_methods",
            "uncovered_focal_methods", "coverage_score", "focal_methods", "covered_methods",
            "uncovered_methods", "atomic_blocks_analysis", "evaluation_algorithm"
        ]
        for key in expected_keys:
            self.assertIn(key, detailed_results)
        
        self.assertEqual(detailed_results["evaluation_algorithm"], "optimal_coverage_one_to_one")
        
        self.assertIsInstance(detailed_results["atomic_blocks_analysis"], list)
        if detailed_results["atomic_blocks_analysis"]:
            block_analysis = detailed_results["atomic_blocks_analysis"][0]
            self.assertIn("block_index", block_analysis)
            self.assertIn("simplified", block_analysis)
            self.assertIn("candidate_methods", block_analysis)
            self.assertIn("notes", block_analysis)

    def test_pipeline_run_localization_agent(self):
        """Test the pipeline's run_localization_agent method with a petclinic-based test case."""
        
        # Create NL2TestInput using data from test2nl.csv (row 3 - low abstraction pet creation test)
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Ensure that the test validates the persistence and ID generation of a new pet associated with an existing owner by first retrieving an owner entity from the database using the OwnerRepository's findById method, confirming the owner exists, and capturing the initial count of pets. Next, instantiate a new Pet, configure its properties including name, birth date, and type (retrieved via the OwnerRepository's findPetTypes method and selected using EntityUtils), and associate it with the retrieved owner via the owner's addPet method, which only adds the pet if it is new. Verify the pet count increases by one on the owner instance, then persist the updated owner using the OwnerRepository's save method. Re-fetch the same owner from the database to confirm the pet count remains incremented, and finally assert that the newly added pet now has a non-null ID, confirming successful persistence and ID generation. The test uses JUnit 5 for execution and AssertJ for fluent assertions, interacting directly with the real OwnerRepository and Pet-related entities without mocking.",
            project_name="spring-petclinic",
            abstraction_level="low",
        )
        
        # Initialize pipeline
        project_root = Path("./resources/spring-petclinic")
        pipeline = Pipeline(self.analysis, project_root)
        
        # Run preprocessing
        method_searcher, class_searcher = pipeline.run_preprocessing()
        self.assertIsNotNone(method_searcher)
        self.assertIsNotNone(class_searcher)
        
        # Decompose natural language
        grammatical_blocks = pipeline.decompose_natural_language(nl2_input.description)
        self.assertIsInstance(grammatical_blocks, list)
        self.assertGreater(len(grammatical_blocks), 0)
        
        # Convert to atomic blocks
        atomic_blocks = [
            AtomicBlock.from_grammatical_block(block) for block in grammatical_blocks
        ]
        self.assertIsInstance(atomic_blocks, list)
        self.assertEqual(len(atomic_blocks), len(grammatical_blocks))
        
        # Run localization agent
        usage_tracker.start()
        localized_blocks, comments = pipeline.run_localization_agent(nl2_input, atomic_blocks)
        prices = usage_tracker.stop()
        
        # Verify outputs
        self.assertIsInstance(localized_blocks, list)
        self.assertIsInstance(comments, str)
        self.assertEqual(len(localized_blocks), len(atomic_blocks))
        
        # Verify that atomic blocks have been enhanced with candidate methods
        for i, block in enumerate(localized_blocks):
            self.assertIsInstance(block, AtomicBlock)
            self.assertEqual(block.order, i)
            # Check that candidate methods have been added (localization should enhance the blocks)
            self.assertIsInstance(block.candidate_methods, list)
        
        pretty_print("Localized blocks", localized_blocks)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_pipeline_run_localization_evaluation_pipeline(self):
        """Test the complete run_localization_evaluation_pipeline method with a petclinic-based test case."""
        
        # Create NL2TestInput using data from test2nl.csv (row 6 - low abstraction vet list test)
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.vet.VetControllerTests",
            method_signature="testShowResourcesVetList()",
            description="Ensure that the veterinarian controller correctly renders a JSON list of veterinarians by first configuring the VetRepository dependency as a Mockito mock during test setup to return two predefined veterinarian objects for both its unpaginated findAll method and its paginated findAll method (when invoked with any Pageable argument), then using the auto-wired MockMvc instance to perform an HTTP GET request to the /vets endpoint with an Accept header specifying JSON media type, and verifying that the response returns a 200 OK status, confirms the content type as JSON, and validates through JsonPath assertions that the first element in the vetList array of the response body contains an identifier matching the expected value for the initial veterinarian instance, all implemented using JUnit 5 for test lifecycle management, Spring Boot Test's @WebMvcTest for web layer testing configuration, Mockito for repository behavior stubbing via @MockitoBean, and Spring MVC Test's MockMvc framework for request execution and response validation with its built-in status, content, and jsonPath matchers.",
            project_name="spring-petclinic",
            abstraction_level=AbstractionLevel("low"),
        )
        
        # Initialize pipeline
        project_root = Path("./resources/spring-petclinic")
        pipeline = Pipeline(self.analysis, project_root)
        
        # Run the complete evaluation pipeline
        usage_tracker.start()
        localized_blocks, coverage_score = pipeline.run_localization_evaluation_pipeline(nl2_input)
        prices = usage_tracker.stop()
        
        # Verify outputs
        self.assertIsInstance(localized_blocks, list)
        self.assertIsInstance(coverage_score, float)
        self.assertGreaterEqual(coverage_score, 0.0)
        self.assertLessEqual(coverage_score, 1.0)
        self.assertGreater(len(localized_blocks), 0)
        
        # Verify that atomic blocks have been enhanced
        for i, block in enumerate(localized_blocks):
            self.assertIsInstance(block, AtomicBlock)
            self.assertEqual(block.order, i)
            self.assertIsInstance(block.candidate_methods, list)
        
        pretty_print("Localized blocks from evaluation pipeline", localized_blocks)
        pretty_print("Coverage score", coverage_score)
        pretty_print("Token usage", prices)
