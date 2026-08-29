from bs4 import BeautifulSoup

from sakura.utils.coverage.individual_test_coverage import IndividualTestCoverage


def test_extract_branch_lines_returns_three_empty_lists_without_source() -> None:
    soup = BeautifulSoup("<html><body>No source block</body></html>", "html.parser")

    extract_branch_lines = getattr(
        IndividualTestCoverage,
        "_IndividualTestCoverage__extract_branch_lines_from_html",
    )
    branch_lines = extract_branch_lines(soup)

    assert branch_lines == ([], [], [])
