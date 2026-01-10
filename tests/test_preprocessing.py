import pytest

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.preprocessing.embedders import HttpEmbedder, OllamaEmbedder
from nltest.nl2test.preprocessing.extractors import MethodSnippetExtractor
from nltest.nl2test.preprocessing.indexers import (
    ClassIndexer,
    MethodIndexer,
    ProjectIndexer,
)
from nltest.nl2test.preprocessing.searchers import (
    ClassSearcher,
    MethodSearcher,
    ProjectSearcher,
)
from nltest.nl2test.preprocessing.vector_stores import MethodVectorStore
from nltest.utils.config import Config
from nltest.utils.pretty.prints import pretty_print


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
