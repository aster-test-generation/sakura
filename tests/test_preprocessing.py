import pytest
from cldk.analysis.java import JavaAnalysis

from sakura.nl2test.preprocessing.embedders import HttpEmbedder, OllamaEmbedder
from sakura.nl2test.preprocessing.extractors import (
    ClassSnippetExtractor,
    MethodSnippetExtractor,
)
from sakura.nl2test.preprocessing.indexers import (
    ClassIndexer,
    MethodIndexer,
    ProjectIndexer,
)
from sakura.nl2test.preprocessing.searchers import (
    ClassSearcher,
    MethodSearcher,
    ProjectSearcher,
)
from sakura.nl2test.preprocessing.vector_stores import MethodVectorStore
from sakura.utils.config import Config
from sakura.utils.pretty.prints import pretty_print


class TestEmbedding:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_config: Config) -> None:
        self.config = petclinic_config

    def test_http_embedder_embed_query(self) -> None:
        emb_model: str = self.config.get("emb", "model")
        api_url: str = self.config.get("emb", "api_url")
        try:
            api_key: str | None = self.config.get("emb", "api_key")
        except Exception:
            api_key = None

        embedder = HttpEmbedder(
            model_id=emb_model,
            api_url=api_url,
            api_key=api_key,
        )

        assert embedder.dim > 0

        embedding = embedder.embed_query("test query")
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dim
        assert all(isinstance(x, float) for x in embedding)

    def test_http_embedder_embed_documents(self) -> None:
        emb_model: str = self.config.get("emb", "model")
        api_url: str = self.config.get("emb", "api_url")
        try:
            api_key: str | None = self.config.get("emb", "api_key")
        except Exception:
            api_key = None

        embedder = HttpEmbedder(
            model_id=emb_model,
            api_url=api_url,
            api_key=api_key,
        )

        docs = ["first document", "second document", "third document"]
        embeddings = embedder.embed_documents(docs)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(docs)
        for emb in embeddings:
            assert len(emb) == embedder.dim


class TestMethodSearch:
    @pytest.fixture(autouse=True)
    def _inject(
        self, petclinic_analysis: JavaAnalysis, petclinic_config: Config
    ) -> None:
        self.analysis = petclinic_analysis
        self.config = petclinic_config

    def test_method_vector_store_single_method_ollama(self) -> None:
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "getPet(java.lang.Integer)"

        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        assert method_details is not None

        emb_model: str = self.config.get("emb", "model")

        embedder = OllamaEmbedder(model_id=emb_model)
        vector_store = MethodVectorStore(embedder)
        method_snippet = MethodSnippetExtractor(self.analysis).get_method_snippet(
            qualified_class_name, method_signature
        )
        assert method_snippet is not None
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

    def test_method_vector_store_single_method_http(self) -> None:
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "getPet(java.lang.Integer)"

        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        assert method_details is not None

        emb_model: str = self.config.get("emb", "model")
        api_url: str = self.config.get("emb", "api_url")
        try:
            api_key: str | None = self.config.get("emb", "api_key")
        except Exception:
            api_key = None

        embedder = HttpEmbedder(
            model_id=emb_model,
            api_url=api_url,
            api_key=api_key,
        )
        vector_store = MethodVectorStore(embedder)
        method_snippet = MethodSnippetExtractor(self.analysis).get_method_snippet(
            qualified_class_name, method_signature
        )
        assert method_snippet is not None
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
        assert len(search_results) == num_similar

    def test_method_vector_store(self) -> None:
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
        assert len(search_results) == num_similar

        num_similar = 3
        desc_substr = "Get the owner from repository with ID 6 and make sure it exists."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Get the owner by ID",
            f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById",
        )

        desc_substr = "Get the owner from the repository return."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Get the owner from repository return",
            f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get",
        )

        desc_substr = "Create a pet."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Create a pet by ID",
            f"FOUND: {search_results}\nACTUAL: default Pet constructor",
        )

        desc_substr = "Add the pet to the owner."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Add pet to owner", f"FOUND: {search_results}\nACTUAL: Owner.addPet"
        )

        desc_substr = "Save the owner to the repository."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Save the owner",
            f"FOUND: {search_results}\nACTUAL: OwnerRepository.save (inherited from JpaRepository library class)",
        )

        desc_substr = "Again, get the owner from the repository with ID 6"
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Get the owner by ID",
            f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById",
        )

        desc_substr = "Get the owner from the repository return."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Get the owner from repository return",
            f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get",
        )

        desc_substr = "Check that the total count of pets in the owner is correct."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Get all pets and check size",
            f"FOUND: {search_results}\nACTUAL: Owner.getPets",
        )

        desc_substr = "Get the added pet from the owner."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print(
            "Get pet in owner", f"FOUND: {search_results}\nACTUAL: Owner.getPet"
        )

        desc_substr = "Get the pet id and check it is non-null."
        search_results = method_searcher.find_similar(desc_substr, k=num_similar)
        pretty_print("Get pet id", f"FOUND: {search_results}\nACTUAL: Pet.getId")


class TestProjectSearch:
    @pytest.fixture(autouse=True)
    def _inject(
        self, petclinic_analysis: JavaAnalysis, petclinic_config: Config
    ) -> None:
        self.analysis = petclinic_analysis
        self.config = petclinic_config

    def test_proj_vector_store_single_search(self) -> None:
        proj_searcher: ProjectSearcher = ProjectIndexer(self.analysis).build_index()
        class_searcher: ClassSearcher = ClassIndexer(self.analysis).build_index()

        num_similar = 5
        desc_substr = "Get the owner from repository with ID 6 and make sure it exists."

        method_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        class_results = proj_searcher.find_classes_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get the owner by ID",
            f"METHOD FOUND: {method_results}\nACTUAL: OwnerRepository.findById",
        )
        pretty_print(
            "Get the owner by ID",
            f"CLASS FOUND: {class_results}\nACTUAL: OwnerRepository",
        )

        class_results = class_searcher.find_similar_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get the owner by ID",
            f"CLASS FOUND: {class_results}\nACTUAL: OwnerRepository",
        )

    def test_proj_vector_store_complete(self) -> None:
        proj_searcher: ProjectSearcher = ProjectIndexer(self.analysis).build_index()

        num_similar = 3
        desc_substr = "Get the owner from repository with ID 6 and make sure it exists."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get the owner by ID",
            f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById",
        )

        desc_substr = "Get the owner from the repository return."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get the owner from repository return",
            f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get",
        )

        desc_substr = "Create a pet."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        search_results = proj_searcher.find_classes_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Create a pet by ID",
            f"FOUND: {search_results}\nACTUAL: default Pet constructor",
        )

        desc_substr = "Add the pet to the owner."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Add pet to owner", f"FOUND: {search_results}\nACTUAL: Owner.addPet"
        )

        desc_substr = "Save the owner to the repository."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        search_results = proj_searcher.find_classes_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Save the owner",
            f"FOUND: {search_results}\nACTUAL: OwnerRepository.save (inherited from JpaRepository library class)",
        )

        desc_substr = "Again, get the owner from the repository with ID 6"
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get the owner by ID",
            f"FOUND: {search_results}\nACTUAL: OwnerRepository.findById",
        )

        desc_substr = "Get the owner from the repository return."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get the owner from repository return",
            f"FOUND: {search_results}\nACTUAL: Optional<Owner>.get",
        )

        desc_substr = "Check that the total count of pets in the owner is correct."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get all pets and check size",
            f"FOUND: {search_results}\nACTUAL: Owner.getPets",
        )

        desc_substr = "Get the added pet from the owner."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print(
            "Get pet in owner", f"FOUND: {search_results}\nACTUAL: Owner.getPet"
        )

        desc_substr = "Get the pet id and check it is non-null."
        search_results = proj_searcher.find_methods_in_range(
            desc_substr, i=1, j=num_similar
        )
        pretty_print("Get pet id", f"FOUND: {search_results}\nACTUAL: Pet.getId")


class TestClassSnippetExtractor:
    """Tests for ClassSnippetExtractor to verify all application classes are extracted."""

    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis: JavaAnalysis) -> None:
        self.analysis = petclinic_analysis
        self.extractor = ClassSnippetExtractor(petclinic_analysis)

    def test_excludes_test_directory_classes(self) -> None:
        """Verify that classes in src/test/java are excluded when exclude_test_dirs=True."""
        snippets = self.extractor.get_class_snippets(exclude_test_dirs=True)

        for snippet in snippets:
            java_file = self.analysis.get_java_file(snippet.declaring_class_name)
            assert "src/test/java" not in java_file, (
                f"Test class {snippet.declaring_class_name} should be excluded"
            )

    def test_includes_all_application_classes_with_visible_methods(self) -> None:
        """Verify that all application classes with visible methods are included."""
        snippets = self.extractor.get_class_snippets(exclude_test_dirs=True)
        extracted_classes = {s.declaring_class_name for s in snippets}

        expected_app_classes = {
            "org.springframework.samples.petclinic.PetClinicApplication",
            "org.springframework.samples.petclinic.PetClinicRuntimeHints",
            "org.springframework.samples.petclinic.model.BaseEntity",
            "org.springframework.samples.petclinic.model.NamedEntity",
            "org.springframework.samples.petclinic.model.Person",
            "org.springframework.samples.petclinic.owner.Owner",
            "org.springframework.samples.petclinic.owner.OwnerController",
            "org.springframework.samples.petclinic.owner.Pet",
            "org.springframework.samples.petclinic.owner.PetController",
            "org.springframework.samples.petclinic.owner.PetType",
            "org.springframework.samples.petclinic.owner.PetTypeFormatter",
            "org.springframework.samples.petclinic.owner.PetValidator",
            "org.springframework.samples.petclinic.owner.Visit",
            "org.springframework.samples.petclinic.owner.VisitController",
            "org.springframework.samples.petclinic.vet.Specialty",
            "org.springframework.samples.petclinic.vet.Vet",
            "org.springframework.samples.petclinic.vet.VetController",
            "org.springframework.samples.petclinic.vet.Vets",
            "org.springframework.samples.petclinic.system.CacheConfiguration",
            "org.springframework.samples.petclinic.system.CrashController",
            "org.springframework.samples.petclinic.system.WebConfiguration",
            "org.springframework.samples.petclinic.system.WelcomeController",
        }

        for expected_class in expected_app_classes:
            if self.analysis.get_class(expected_class) is not None:
                from sakura.utils.analysis import Reachability

                visible_methods = Reachability(self.analysis).get_visible_class_methods(
                    expected_class
                )
                if visible_methods:
                    assert expected_class in extracted_classes, (
                        f"Expected class {expected_class} with visible methods not found"
                    )

    def test_no_duplicate_classes(self) -> None:
        """Verify each class appears exactly once in the results."""
        snippets = self.extractor.get_class_snippets(exclude_test_dirs=True)
        class_names = [s.declaring_class_name for s in snippets]
        assert len(class_names) == len(set(class_names)), (
            "Duplicate classes found in extraction results"
        )


class TestMethodSnippetExtractor:
    """Tests for MethodSnippetExtractor to verify all application methods are extracted."""

    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis: JavaAnalysis) -> None:
        self.analysis = petclinic_analysis
        self.extractor = MethodSnippetExtractor(petclinic_analysis)

    def test_excludes_test_directory_methods(self) -> None:
        """Verify that methods in src/test/java are excluded when exclude_test_dirs=True."""
        snippets = self.extractor.get_project_snippets(exclude_test_dirs=True)

        for snippet in snippets:
            java_file = self.analysis.get_java_file(snippet.containing_class_name)
            assert "src/test/java" not in java_file, (
                f"Test method {snippet.method_signature} in {snippet.containing_class_name} "
                "should be excluded"
            )

    def test_inherited_methods_counted_per_class(self) -> None:
        """Verify that inherited methods are counted for each class they appear in.

        For example, BaseEntity.getId() should appear for Owner, Pet, etc.
        """
        snippets = self.extractor.get_project_snippets(exclude_test_dirs=True)

        method_class_pairs: set[tuple[str, str]] = set()
        for snippet in snippets:
            pair = (snippet.method_signature, snippet.containing_class_name)
            method_class_pairs.add(pair)

        inherited_method_sig = "getId()"
        classes_with_getId = [
            containing_class
            for sig, containing_class in method_class_pairs
            if sig == inherited_method_sig
        ]

        expected_classes_with_getId = {
            "org.springframework.samples.petclinic.model.BaseEntity",
            "org.springframework.samples.petclinic.model.NamedEntity",
            "org.springframework.samples.petclinic.model.Person",
            "org.springframework.samples.petclinic.owner.Owner",
            "org.springframework.samples.petclinic.owner.Pet",
            "org.springframework.samples.petclinic.owner.PetType",
            "org.springframework.samples.petclinic.owner.Visit",
            "org.springframework.samples.petclinic.vet.Specialty",
            "org.springframework.samples.petclinic.vet.Vet",
        }

        for expected_class in expected_classes_with_getId:
            if self.analysis.get_class(expected_class) is not None:
                assert expected_class in classes_with_getId, (
                    f"Inherited method getId() should be counted for {expected_class}"
                )

    def test_class_own_methods_included(self) -> None:
        """Verify that each class's own methods are included in its snippets."""
        owner_class = "org.springframework.samples.petclinic.owner.Owner"
        snippets = self.extractor.get_class_snippets(
            owner_class, exclude_test_dirs=True
        )

        method_sigs = {s.method_signature for s in snippets}

        expected_own_methods = {
            "getAddress()",
            "setAddress(String)",
            "getCity()",
            "setCity(String)",
            "getTelephone()",
            "setTelephone(String)",
            "getPets()",
            "addPet(Pet)",
            "getPet(String)",
            "getPet(Integer)",
            "getPet(String, boolean)",
            "addVisit(Integer, Visit)",
            "toString()",
        }

        for expected in expected_own_methods:
            assert expected in method_sigs, (
                f"Expected method {expected} not found for Owner class"
            )

    def test_inherited_methods_included_for_subclass(self) -> None:
        """Verify that inherited methods from parent classes are included for subclasses."""
        owner_class = "org.springframework.samples.petclinic.owner.Owner"
        snippets = self.extractor.get_class_snippets(
            owner_class, exclude_test_dirs=True
        )

        method_sigs = {s.method_signature for s in snippets}

        expected_inherited_methods = {
            "getId()",
            "setId(Integer)",
            "isNew()",
            "getFirstName()",
            "setFirstName(String)",
            "getLastName()",
            "setLastName(String)",
        }

        for expected in expected_inherited_methods:
            assert expected in method_sigs, (
                f"Expected inherited method {expected} not found for Owner class"
            )

    def test_no_duplicate_methods_per_class(self) -> None:
        """Verify that each method appears at most once per containing class."""
        snippets = self.extractor.get_project_snippets(exclude_test_dirs=True)

        method_class_pairs = [
            (s.method_signature, s.containing_class_name) for s in snippets
        ]
        assert len(method_class_pairs) == len(set(method_class_pairs)), (
            "Duplicate method-class pairs found in extraction results"
        )

    def test_method_count_reflects_inheritance(self) -> None:
        """Verify method counts correctly reflect inheritance accumulation.

        Classes further down the hierarchy should have more methods due to
        accumulated inheritance.
        """
        base_entity = "org.springframework.samples.petclinic.model.BaseEntity"
        named_entity = "org.springframework.samples.petclinic.model.NamedEntity"
        person = "org.springframework.samples.petclinic.model.Person"
        owner = "org.springframework.samples.petclinic.owner.Owner"

        base_snippets = self.extractor.get_class_snippets(
            base_entity, exclude_test_dirs=True
        )
        named_snippets = self.extractor.get_class_snippets(
            named_entity, exclude_test_dirs=True
        )
        person_snippets = self.extractor.get_class_snippets(
            person, exclude_test_dirs=True
        )
        owner_snippets = self.extractor.get_class_snippets(
            owner, exclude_test_dirs=True
        )

        assert len(base_snippets) > 0, "BaseEntity should have methods"
        assert len(named_snippets) > len(base_snippets), (
            "NamedEntity should have more methods than BaseEntity due to inheritance"
        )
        assert len(person_snippets) > len(base_snippets), (
            "Person should have more methods than BaseEntity due to inheritance"
        )
        assert len(owner_snippets) > len(person_snippets), (
            "Owner should have more methods than Person due to additional own methods"
        )
