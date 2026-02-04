from typing import List

import pytest

from sakura.test2nl.extractors import (
    ClassExtractor,
    FieldDeclarationExtractor,
    MethodExtractor,
)
from sakura.test2nl.model.models import (
    AbstractionLevel,
    Test2NLEntry,
    TestDescriptionInfo,
)
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.pretty.prints import pretty_print


class TestDescriptionGeneration:
    def test_desc_high_abs_one_method(
        self, petclinic_desc_generator, petclinic_data_manager
    ):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        abstraction_level = AbstractionLevel.HIGH
        desc_generator = petclinic_desc_generator
        test_description_info: TestDescriptionInfo | None = (
            desc_generator.generate_for_method(
                method_signature, qualified_class_name, abstraction_level
            )
        )
        assert test_description_info is not None, "LLM generation failed..."

        test_descriptions = [test_description_info]
        petclinic_data_manager.save("descriptions.json", test_descriptions)

    def test_desc_medium_abs_one_method(
        self, petclinic_desc_generator, petclinic_data_manager
    ):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        abstraction_level = AbstractionLevel.MEDIUM
        desc_generator = petclinic_desc_generator
        test_description_info: TestDescriptionInfo | None = (
            desc_generator.generate_for_method(
                method_signature, qualified_class_name, abstraction_level
            )
        )
        assert test_description_info is not None, "LLM generation failed..."

        test_descriptions = [test_description_info]
        petclinic_data_manager.save("descriptions.json", test_descriptions)

    def test_desc_low_abs_one_method(
        self, petclinic_desc_generator, petclinic_data_manager
    ):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        abstraction_level = AbstractionLevel.LOW
        desc_generator = petclinic_desc_generator
        test_description_info: TestDescriptionInfo | None = (
            desc_generator.generate_for_method(
                method_signature, qualified_class_name, abstraction_level
            )
        )
        assert test_description_info is not None, "LLM generation failed..."

        test_descriptions = [test_description_info]
        petclinic_data_manager.save("descriptions.json", test_descriptions)

    def test_desc_all_abs_one_method(
        self, petclinic_desc_generator, petclinic_data_manager, petclinic_paths
    ):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        desc_generator = petclinic_desc_generator
        test_descriptions = []
        for abs_level in AbstractionLevel:
            test_description_info = desc_generator.generate_for_method(
                method_signature, qualified_class_name, abs_level
            )
            if test_description_info:
                test_descriptions.append(test_description_info)
        assert test_descriptions, "Descriptions could not be generated..."

        data_manager = petclinic_data_manager
        test2nl_entries: List[Test2NLEntry] = []
        entry_id = 1
        for test_description_info in test_descriptions:
            test_description_info.id = entry_id
            entry = Test2NLEntry.from_test_description_info(
                test_description_info, petclinic_paths.project_name
            )
            test2nl_entries.append(entry)
            entry_id += 1

        data_manager.save("descriptions.json", test_descriptions, format="json")
        data_manager.save("test2nl.csv", test2nl_entries, format="csv")
        descriptions_path = data_manager.base_dir / "descriptions.json"
        csv_path = data_manager.base_dir / "test2nl.csv"
        assert descriptions_path.exists(), "Descriptions file was not created"
        assert csv_path.exists(), "Test2NL CSV file was not created"

    def test_test2nl_load(self, petclinic_data_manager, petclinic_paths):
        csv_path = petclinic_paths.project_output_dir / "test2nl.csv"
        if not csv_path.exists():
            pytest.skip("test2nl.csv not found; run test_desc_all_abs_one_method first")

        test2nl: List[Test2NLEntry] = petclinic_data_manager.load(
            "test2nl.csv", Test2NLEntry, format="csv"
        )
        assert len(test2nl) > 0, "Test2NL is empty..."

        single_entry = test2nl[0]
        pretty_print("Test2NL Entry", single_entry)

        single_description = single_entry.description
        pretty_print("Test2NL Description", single_description)

    def test_description_multiple_focal(
        self,
        petclinic_analysis,
        petclinic_desc_generator,
        petclinic_data_manager,
        petclinic_paths,
    ):
        analysis = petclinic_analysis
        complicated_tests = CommonAnalysis(analysis).get_complicated_focal_tests()
        total_count = CommonAnalysis(analysis).get_complicated_focal_tests_count()

        pretty_print("Complicated focal tests by class", complicated_tests)
        pretty_print("Total number of complicated focal test methods", total_count)

        assert isinstance(complicated_tests, dict), "Should return a dictionary"
        assert total_count >= 0, "Count should be non-negative"

        desc_generator = petclinic_desc_generator
        test_descriptions: List[TestDescriptionInfo] = []
        entry_id = 1

        for test_class, methods in complicated_tests.items():
            pretty_print(
                f"Processing class {test_class} with {len(methods)} methods", methods
            )

            for method_signature in methods:
                for abs_level in AbstractionLevel:
                    test_description_info = desc_generator.generate_for_method(
                        method_signature, test_class, abs_level
                    )
                    if test_description_info:
                        test_description_info.id = entry_id
                        test_descriptions.append(test_description_info)
                        entry_id += 1

            if entry_id > 15:
                break

        assert test_descriptions, "No descriptions could be generated..."
        pretty_print(
            f"Generated {len(test_descriptions)} test descriptions",
            len(test_descriptions),
        )

        entries: List[Test2NLEntry] = []
        for test_description_info in test_descriptions:
            entry = Test2NLEntry.from_test_description_info(
                test_description_info, petclinic_paths.project_name
            )
            entries.append(entry)

        data_manager = petclinic_data_manager
        data_manager.save(
            "descriptions.json", test_descriptions, format="json", mode="append"
        )
        data_manager.save("test2nl.csv", entries, format="csv", mode="append")

        print(
            f"Successfully saved {len(test_descriptions)} descriptions and {len(entries)} Test2NL entries"
        )


class TestDescriptionContextExtraction:
    """
    Tests for extractors in src/sakura/test2nl/extractors/
    Using Spring PetClinic resources.
    """

    def test_field_declaration_extractor(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting fields from Owner class.
        Owner class has private fields: address, city, telephone, pets.
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        class_details = petclinic_analysis.get_class(qualified_class_name)
        assert class_details is not None, f"Class {qualified_class_name} not found"

        # Find 'address' field
        address_field = next(
            (
                f
                for f in class_details.field_declarations
                if f.variables and "address" in f.variables
            ),
            None,
        )
        assert address_field is not None, "Field 'address' not found in Owner class"

        extractor = FieldDeclarationExtractor(petclinic_analysis, application_classes)
        field_info = extractor.extract(address_field)

        assert field_info is not None
        assert field_info.variables is not None
        assert "address" in field_info.variables
        assert field_info.type == "java.lang.String"
        assert field_info.modifiers is not None
        assert "private" in field_info.modifiers
        # Annotations are strings containing the full annotation (e.g. "@Column(name = \"address\")")
        assert field_info.annotations is not None
        assert any("Column" in a for a in field_info.annotations)
        assert any("NotBlank" in a for a in field_info.annotations)

    def test_method_extractor_test_method(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting a test method: ClinicServiceTests.shouldInsertPetIntoDatabaseAndGenerateId
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        extractor = MethodExtractor(petclinic_analysis, application_classes)
        method_context = extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        assert method_context.method_signature == method_signature
        assert method_context.is_getter_or_setter is False
        assert method_context.code is not None
        assert "shouldInsertPetIntoDatabaseAndGenerateId" in method_context.code

        # Check call sites - the method calls findById, getPets, save, getPet, addPet, etc.
        call_methods = [cs.method_name for cs in method_context.call_sites]
        assert "findById" in call_methods
        assert "save" in call_methods
        assert "getPet" in call_methods

        # Verify assertions are detected (uses assertThat from AssertJ)
        assert any(cs.is_assertion for cs in method_context.call_sites)

    def test_method_extractor_domain_method(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting a domain method: Owner.addPet(Pet pet)
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "addPet(org.springframework.samples.petclinic.owner.Pet)"

        extractor = MethodExtractor(petclinic_analysis, application_classes)
        method_context = extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        assert method_context.method_signature == method_signature
        assert method_context.is_getter_or_setter is False
        assert method_context.code is not None

        # It calls getPets().add(pet) and pet.isNew()
        call_methods = [cs.method_name for cs in method_context.call_sites]
        assert "getPets" in call_methods
        assert "add" in call_methods

    def test_method_extractor_getter(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting a getter: Owner.getAddress()
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        method_signature = "getAddress()"

        extractor = MethodExtractor(petclinic_analysis, application_classes)
        method_context = extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        assert method_context.method_signature == method_signature
        assert method_context.is_getter_or_setter is True
        # Code should be None for getter/setter even if complete_methods=True
        assert method_context.code is None

    def test_referenced_class_extractor(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting referenced class info: Owner (no filter)
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"

        extractor = ClassExtractor(petclinic_analysis, application_classes)
        ref_class = extractor.extract(qualified_class_name, complete_methods=False)

        assert ref_class.qualified_class_name == qualified_class_name
        assert ref_class.simple_class_name == "Owner"
        assert ref_class.extends is not None
        assert "org.springframework.samples.petclinic.model.Person" in ref_class.extends

        # Annotations are strings (e.g. "@Entity", "@Table(name = \"owners\")")
        assert ref_class.annotations is not None
        assert any("Entity" in a for a in ref_class.annotations)
        assert any("Table" in a for a in ref_class.annotations)

        # Check methods exist (no filter = all methods included)
        assert ref_class.relevant_class_methods is not None
        method_sigs = [m.method_signature for m in ref_class.relevant_class_methods]
        assert "getAddress()" in method_sigs
        assert "setAddress(java.lang.String)" in method_sigs
        assert "addPet(org.springframework.samples.petclinic.owner.Pet)" in method_sigs

        # Owner class has no public fields, so field_declarations should be empty or None
        # The extractor filters for public fields only
        if ref_class.field_declarations:
            for fd in ref_class.field_declarations:
                assert fd.modifiers is not None
                assert "public" in fd.modifiers

    def test_referenced_class_extractor_with_filter(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting referenced class with called_method_names filter.
        Only methods in the filter (plus constructors) should be included.
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"

        extractor = ClassExtractor(petclinic_analysis, application_classes)
        called_methods = {"getAddress", "addPet"}
        ref_class = extractor.extract(
            qualified_class_name,
            complete_methods=False,
            called_method_names=called_methods,
        )

        assert ref_class.relevant_class_methods is not None
        method_sigs = [m.method_signature for m in ref_class.relevant_class_methods]

        # Filtered methods should be included
        assert "getAddress()" in method_sigs
        assert "addPet(org.springframework.samples.petclinic.owner.Pet)" in method_sigs

        # Methods not in filter should be excluded
        assert "setAddress(java.lang.String)" not in method_sigs
        assert "getCity()" not in method_sigs

    def test_referenced_class_extractor_constructors_always_included(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test that constructors are always included regardless of filter.
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = (
            "org.springframework.samples.petclinic.owner.PetController"
        )

        extractor = ClassExtractor(petclinic_analysis, application_classes)
        # Filter with a method that doesn't exist - should still include constructors
        called_methods = {"nonExistentMethod"}
        ref_class = extractor.extract(
            qualified_class_name,
            complete_methods=False,
            called_method_names=called_methods,
        )

        assert ref_class.relevant_class_methods is not None
        method_sigs = [m.method_signature for m in ref_class.relevant_class_methods]

        # Constructors should always be included (Pet class has a constructor)
        has_constructor = any(
            sig.startswith("PetController") or sig.startswith("<init>")
            for sig in method_sigs
        )
        assert has_constructor, f"Constructor not found in: {method_sigs}"

    def test_referenced_class_extractor_empty_filter(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test extracting with empty filter - only constructors should be included.
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"

        extractor = ClassExtractor(petclinic_analysis, application_classes)
        ref_class = extractor.extract(
            qualified_class_name,
            complete_methods=False,
            called_method_names=set(),  # Empty filter
        )

        # Class metadata should still be present
        assert ref_class.qualified_class_name == qualified_class_name
        assert ref_class.simple_class_name == "Owner"
        assert ref_class.annotations is not None

        # With empty filter, only constructors should be included
        if ref_class.relevant_class_methods:
            for method in ref_class.relevant_class_methods:
                method_name = method.method_signature.split("(")[0]
                assert method_name == "Owner" or method_name == "<init>", (
                    f"Non-constructor method {method.method_signature} "
                    "should not be included with empty filter"
                )

    def test_entity_utils_call_site_is_helper(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test that EntityUtils.getById call site has is_helper=True.
        EntityUtils is a test utility class (in src/test/java) but not an application class.
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        extractor = MethodExtractor(petclinic_analysis, application_classes)
        method_context = extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        # Find the getById call site (called on EntityUtils)
        entity_utils_class = "org.springframework.samples.petclinic.service.EntityUtils"
        get_by_id_calls = [
            cs
            for cs in method_context.call_sites
            if cs.method_name == "getById" and cs.receiver_type == entity_utils_class
        ]

        assert len(get_by_id_calls) > 0, (
            f"Expected at least one getById call on EntityUtils, "
            f"found call sites: {[(cs.method_name, cs.receiver_type) for cs in method_context.call_sites]}"
        )

        # Verify is_helper is True for EntityUtils calls
        for call_site in get_by_id_calls:
            assert call_site.is_helper is True, (
                f"Expected is_helper=True for EntityUtils.getById call, "
                f"got is_helper={call_site.is_helper}"
            )

    def test_entity_utils_is_test_utility_class(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test that EntityUtils is categorized as a test utility class,
        not an application class.
        """
        application_classes, test_utility_classes = petclinic_categorized_classes
        entity_utils_class = "org.springframework.samples.petclinic.service.EntityUtils"

        # EntityUtils should NOT be in application_classes
        assert entity_utils_class not in application_classes, (
            "EntityUtils should not be an application class"
        )

        # EntityUtils should be in test_utility_classes
        assert entity_utils_class in test_utility_classes, (
            "EntityUtils should be a test utility class"
        )

        # Verify the class exists in the analysis
        class_details = petclinic_analysis.get_class(entity_utils_class)
        assert class_details is not None, "EntityUtils class not found in analysis"

    def test_application_class_call_site_not_helper(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """
        Test that calls to application classes have is_helper=False.
        Owner is an application class, so calls to Owner methods should not be helpers.
        """
        application_classes, _ = petclinic_categorized_classes
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        extractor = MethodExtractor(petclinic_analysis, application_classes)
        method_context = extractor.extract(
            qualified_class_name, method_signature, complete_methods=True
        )

        # Find calls to Owner methods (an application class)
        owner_class = "org.springframework.samples.petclinic.owner.Owner"
        owner_calls = [
            cs for cs in method_context.call_sites if cs.receiver_type == owner_class
        ]

        assert len(owner_calls) > 0, (
            f"Expected at least one call on Owner class, "
            f"found call sites: {[(cs.method_name, cs.receiver_type) for cs in method_context.call_sites]}"
        )

        # Verify is_helper is False for application class calls
        for call_site in owner_calls:
            assert call_site.is_helper is False, (
                f"Expected is_helper=False for Owner.{call_site.method_name} call, "
                f"got is_helper={call_site.is_helper}"
            )
