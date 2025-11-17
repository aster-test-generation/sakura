from langchain_core.tools import StructuredTool

from nltest.nl2test.generation.composition.orchestrators import (
    GherkinCompositionOrchestrator,
)
from nltest.nl2test.models import NL2TestInput, LocalizedScenario, AbstractionLevel
from nltest.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.pretty.prints import pretty_print


class TestCompositionAgent:
    def test_composition_system_prompt_formatting(self):
        """Verify system prompt formatting for composition Gherkin with placeholders."""
        prompt = LoadPrompt.load_prompt(
            "composition_agent_gherkin.jinja2", PromptFormat.JINJA2, "system"
        )

        # parallelizable True path
        iters_true = 4
        rendered_true = prompt.format(parallelizable=True, max_iters=iters_true)
        pretty_print("Parallelizable prompt", rendered_true)
        expected_true = (
            f"You must complete within at most {iters_true} tool invocation(s)."
        )
        assert expected_true in rendered_true
        assert "You may parallelize metadata lookups" in rendered_true
        assert "Use sequential calls." not in rendered_true

        # parallelizable False path
        iters_false = 6
        rendered_false = prompt.format(parallelizable=False, max_iters=iters_false)
        pretty_print("Not parallelizable prompt", rendered_false)
        expected_false = (
            f"You must complete within at most {iters_false} tool invocation(s)."
        )
        assert expected_false in rendered_false
        assert "Use sequential calls." in rendered_false
        assert "You may parallelize metadata lookups" not in rendered_false

        # duplicate_tools path — mirror orchestrator formatting
        # Create simple tools using LangChain's StructuredTool (a BaseTool subclass)
        dup_tools = [
            StructuredTool.from_function(
                func=lambda: None, name="view_test_code", description=""
            ),
            StructuredTool.from_function(
                func=lambda: None,
                name="compile_and_execute_tests",
                description="",
            ),
        ]
        duplicate_tools_str = ", ".join(f"`{t.name}`" for t in dup_tools)

        rendered_with_dups = prompt.format(
            parallelizable=True,
            max_iters=iters_true,
            duplicate_tools=duplicate_tools_str,
        )
        pretty_print("With duplicate_tools", rendered_with_dups)

        assert (
            "You are only allowed to repeat the `view_test_code`, `compile_and_execute_tests` tools"
            in rendered_with_dups
        )
        assert "- Never repeat the same {tool, args} pair" not in rendered_with_dups
        assert "You are only allowed to repeat" not in rendered_true

    def test_composition_chat_prompt_formatting(self):
        """Verify chat prompt formatting for composition Gherkin with placeholders."""
        prompt = LoadPrompt.load_prompt(
            "composition_agent_gherkin.jinja2", PromptFormat.JINJA2, "chat"
        )

        nl_description = "Compose a unit test for PetController update"
        instructions = "Use JUnit 5 and avoid Mockito unless necessary"
        localized_scenario = "{\n  'given': [], 'when': [], 'then': []\n}"

        rendered = prompt.format(
            nl_description=nl_description,
            instructions=instructions,
            localized_scenario=localized_scenario,
        )

        assert nl_description in rendered
        assert instructions in rendered
        assert "CURRENT LOCALIZED SCENARIO" in rendered
        assert "Compose a compilable and runnable Java test" in rendered

    def test_composition_agent_gherkin(self, nl2test_context):
        analysis = nl2test_context.analysis
        method_searcher = MethodIndexer(analysis).build_index()
        class_searcher = ClassIndexer(analysis).build_index()

        project_name = nl2test_context.project_name
        project_root = nl2test_context.resources_dir / project_name

        nl2_input = NL2TestInput(
            id=-1,
            description=(
                "Create a test case that validates the successful processing of a pet update form. "
                "The test begins by setting up the test environment using the `@WebMvcTest` annotation to load the "
                "`PetController` and `PetTypeFormatter` within a Spring MVC test context, disabling the test in native image "
                "and AOT modes via `@DisabledInNativeImage` and `@DisabledInAotMode` respectively. The `OwnerRepository` "
                "dependency of the `PetController` is mocked using `@MockitoBean`, and the `MockMvc` is autowired for "
                "simulating HTTP requests. The setup method stubs the `findPetTypes` method of the mocked `OwnerRepository` "
                "to return a list containing a `PetType` object with a predefined ID and name, and also stubs the `findById` "
                "method to return an `Optional` containing an `Owner` object populated with two `Pet` objects, each having a "
                'unique ID and name. The test then performs a `POST` request to the "/owners/{ownerId}/pets/{petId}/edit" '
                "endpoint, substituting predefined `TEST_OWNER_ID` and `TEST_PET_ID` values into the URL, and including "
                "parameters for the pet's name, type, and birth date. The test uses `MockMvc` to simulate the request and "
                "verifies that the response has a 3xx redirection status code and that the view name is a redirection to the "
                "owner's details page using `andExpect` with `status().is3xxRedirection()` and `view().name()`. JUnit and "
                "Mockito are used for structuring the test and mocking dependencies, while Spring's `MockMvc` and its "
                "associated matchers are used for simulating and asserting on the web layer."
            ),
            project_name=project_name,
            qualified_class_name="org.springframework.samples.petclinic.owner.PetControllerTests",
            method_signature="testProcessUpdateFormSuccess()",
            abstraction_level=AbstractionLevel.LOW,
            is_bdd=False,
        )

        composition_agent = GherkinCompositionOrchestrator(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            project_root=str(project_root),
        )

        localized_scenario_data = {
            "testing_framework": "junit",
            "setup": [
                {
                    "id": 0,
                    "task": "Load Spring MVC test context for PetController and PetTypeFormatter using @WebMvcTest",
                    "uses": "",
                    "produces": "mock_mvc_context",
                    "candidate_methods": [],
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
                    "arg_bindings": [],
                    "comments": "@WebMvcTest is an annotation, not a method. It's used at the class level to configure the Spring application context for testing a Spring MVC controller.",
                    "external": False,
                },
                {
                    "id": 1,
                    "task": "Disable test in native image and AOT modes",
                    "uses": "mock_mvc_context",
                    "produces": "",
                    "candidate_methods": [],
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
                    "arg_bindings": [],
                    "comments": "@DisabledInNativeImage and @DisabledInAotMode are annotations used at the class level to disable tests in specific Spring Boot modes. They are not methods.",
                    "external": False,
                },
                {
                    "id": 2,
                    "task": "Mock OwnerRepository dependency using @MockitoBean",
                    "uses": "mock_mvc_context",
                    "produces": "mock_owner_repository",
                    "candidate_methods": [],
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
                    "arg_bindings": [],
                    "comments": "@MockitoBean is an annotation used to add Mockito mocks to the Spring application context. It's not a method call but a declaration.",
                    "external": False,
                },
                {
                    "id": 3,
                    "task": "Autowire MockMvc for simulating HTTP requests",
                    "uses": "mock_mvc_context",
                    "produces": "mock_mvc",
                    "candidate_methods": [],
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
                    "arg_bindings": [],
                    "comments": "@Autowired is an annotation used for dependency injection. MockMvc is typically injected into the test class, not called as a method.",
                    "external": False,
                },
                {
                    "id": 4,
                    "task": "Stub findPetTypes method of mocked OwnerRepository to return a list with a PetType",
                    "uses": "mock_owner_repository",
                    "produces": "pet_types",
                    "candidate_methods": [
                        {
                            "implementing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "method_signature": "findPetTypes()",
                            "return_type": "java.util.List<org.springframework.samples.petclinic.owner.PetType>",
                        }
                    ],
                    "best_candidate": {
                        "implementing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                        "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                        "method_signature": "findPetTypes()",
                        "return_type": "java.util.List<org.springframework.samples.petclinic.owner.PetType>",
                    },
                    "arg_bindings": [],
                    "comments": "This step involves Mockito's `when().thenReturn()` syntax to stub the `findPetTypes` method.",
                    "external": False,
                },
                {
                    "id": 5,
                    "task": "Stub findById method of mocked OwnerRepository to return an Optional containing an Owner with two Pets",
                    "uses": "mock_owner_repository",
                    "produces": "owner_with_pets",
                    "candidate_methods": [
                        {
                            "implementing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "method_signature": "findById(java.lang.Integer)",
                            "return_type": "java.util.Optional<org.springframework.samples.petclinic.owner.Owner>",
                        }
                    ],
                    "best_candidate": {
                        "implementing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                        "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                        "method_signature": "findById(java.lang.Integer)",
                        "return_type": "java.util.Optional<org.springframework.samples.petclinic.owner.Owner>",
                    },
                    "arg_bindings": [{"arg_name": "id", "arg_value": "TEST_OWNER_ID"}],
                    "comments": "This step involves Mockito's `when().thenReturn()` syntax to stub the `findById` method.",
                    "external": False,
                },
            ],
            "steps": [
                {
                    "given": [],
                    "when": [
                        {
                            "id": 6,
                            "task": "Perform POST request to /owners/{ownerId}/pets/{petId}/edit with pet details",
                            "uses": "mock_mvc, TEST_OWNER_ID, TEST_PET_ID, pet_name, pet_type, pet_birth_date",
                            "produces": "http_response",
                            "candidate_methods": [
                                {
                                    "implementing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                    "method_signature": "processUpdateForm(org.springframework.samples.petclinic.owner.Owner, org.springframework.samples.petclinic.owner.Pet, org.springframework.validation.BindingResult, org.springframework.web.servlet.mvc.support.RedirectAttributes)",
                                    "return_type": "java.lang.String",
                                }
                            ],
                            "best_candidate": {
                                "implementing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                "containing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                "method_signature": "processUpdateForm(org.springframework.samples.petclinic.owner.Owner, org.springframework.samples.petclinic.owner.Pet, org.springframework.validation.BindingResult, org.springframework.web.servlet.mvc.support.RedirectAttributes)",
                                "return_type": "java.lang.String",
                            },
                            "arg_bindings": [
                                {"arg_name": "owner", "arg_value": "owner_with_pets"},
                                {"arg_name": "pet", "arg_value": "pet_details"},
                                {
                                    "arg_name": "result",
                                    "arg_value": "new BindingResult()",
                                },
                                {
                                    "arg_name": "redirectAttributes",
                                    "arg_value": "new RedirectAttributes()",
                                },
                            ],
                            "comments": "This step will use MockMvc.perform(post(...)) to simulate the HTTP POST request. The processUpdateForm method in PetController is the target method for this request.",
                            "external": False,
                        }
                    ],
                    "then": [
                        {
                            "id": 7,
                            "task": "Verify response has 3xx redirection status code",
                            "uses": "http_response",
                            "produces": "",
                            "candidate_methods": [],
                            "best_candidate": {
                                "implementing_class_name": "",
                                "containing_class_name": "",
                                "method_signature": "",
                                "return_type": "",
                            },
                            "arg_bindings": [],
                            "comments": "This step will use MockMvcResultMatchers.status().is3xxRedirection(). This is a static method from Spring Test.",
                            "external": False,
                        },
                        {
                            "id": 8,
                            "task": "Verify view name is a redirection to the owner's details page",
                            "uses": "http_response",
                            "produces": "",
                            "candidate_methods": [],
                            "best_candidate": {
                                "implementing_class_name": "",
                                "containing_class_name": "",
                                "method_signature": "",
                                "return_type": "",
                            },
                            "arg_bindings": [],
                            "comments": "This step will use MockMvcResultMatchers.view().name(). This is a static method from Spring Test.",
                            "external": False,
                        },
                    ],
                }
            ],
            "teardown": [],
        }

        localized_scenario = LocalizedScenario(**localized_scenario_data)

        instructions = "Compose the Java test for this scenario and finalize."
        updated_scenario, final_comments, package, class_name = (
            composition_agent.assign_task(localized_scenario, instructions=instructions)
        )

        assert isinstance(updated_scenario, LocalizedScenario)
        assert isinstance(final_comments, str)

        # Pretty print selected package and class name before assertions
        pretty_print("Selected package", package)
        pretty_print("Selected test class", class_name)

        # Ensure package and class name were selected
        assert package is not None
        assert class_name is not None

        pretty_print("Updated scenario", updated_scenario)
        pretty_print("Final comments", final_comments)
