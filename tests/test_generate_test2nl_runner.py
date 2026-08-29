from scripts.runners import generate_test2nl


def test_default_input_directories_match_repository_resources() -> None:
    assert generate_test2nl.ANALYSIS_DIR == "resources/analysis"
    assert generate_test2nl.ORGANIZED_METHODS_DIR == "resources/filtered_bucketed_tests"
