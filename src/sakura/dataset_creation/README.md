# Dataset Creation Pipeline

This module provides utilities for creating and processing the NL2Test dataset from Java project repositories.

## Execution Order

The pipeline should be executed in the following order:

### 1. Create Hamster Models (`create_hamster_model.py`)

Analyzes source projects using CLDK and generates hamster analysis models containing test class and focal method information.

```
Input:  resources/datasets/<project>/        (Java project repositories)
Output: resources/hamster/<project>/hamster.json
Cache:  resources/analysis/<project>/        (CLDK analysis cache)
```

```bash
python -m sakura.dataset_creation.create_hamster_model
```

### 2. Bucketize Dataset (`bucketize_dataset.py`)

Processes hamster models to categorize tests by focal method count into buckets (1, 2, 3-5, 6-10, >10 focal methods). Filters out abstract classes, interfaces, enums, and annotation declarations.

```
Input:  resources/hamster_models/<project>/hamster.json
Output: resources/bucketed_tests/<project>/nl2test.json
```

```bash
python -m sakura.dataset_creation.bucketize_dataset
```

### 3. Filter by Date (`filter_by_date.py`)

Identifies test methods added to repositories after a specified date using git history and Tree-sitter parsing. Can run independently of steps 1-2.

```
Input:  resources/datasets/<project>/        (Git repositories)
Output: resources/filtered_tests/<project>/nl2test.json
        resources/filtered_tests/summary.json
```

```bash
python -m sakura.dataset_creation.filter_by_date
```

Default cutoff date: `2025-01-31`

### 4. Bucketize Filtered Dataset (`bucketize_filtered_dataset.py`)

Combines bucketed datasets with date-filtered results, producing a dataset that contains only tests added after the cutoff date while preserving the focal method bucketing.

```
Input:  resources/bucketed_tests/<project>/nl2test.json
        resources/filtered_tests/<project>/nl2test.json
Output: resources/filtered_bucketed_tests/<project>/nl2test.json
        resources/filtered_bucketed_tests/summary.json
```

```bash
python -m sakura.dataset_creation.bucketize_filtered_dataset
```

### 5. Verify Dataset Tests (`verify_dataset_tests.py`)

Validates that all tests in the dataset can be found via CLDK analysis. Identifies erroneous tests where `get_method(qualified_class_name, method_signature)` returns falsy, indicating potential mismatches between dataset entries and actual source code.

```
Input:  resources/filtered_bucketed_tests/<project>/nl2test.json
        resources/analysis/<project>/           (CLDK analysis cache)
        resources/datasets/<project>/           (Source projects)
Output: resources/erroneous_tests/<project>/erroneous_tests.json
        resources/erroneous_tests/summary.json
```

```bash
python -m sakura.dataset_creation.verify_dataset_tests
```

### 6. Find Missing Test2NL Entries (`find_missing_test2nl_entries.py`)

Identifies tests in the bucketed dataset that are missing or incomplete in the Test2NL output CSV. A test is considered missing if it either doesn't exist in Test2NL at all, or if it lacks all three required abstraction levels (low, medium, high). Used to recover from incomplete Test2NL generation runs (e.g., due to rate limits or partial failures). Outputs missing tests in NL2TestDataset format for re-processing.

```
Input:  resources/test2nl/filtered_dataset/test2nl.csv
        resources/filtered_bucketed_tests/<project>/nl2test.json
Output: resources/missing_tests/<project>/nl2test.json
        resources/missing_tests/summary.json
```

```bash
python -m sakura.dataset_creation.find_missing_test2nl_entries
```

### 7. Merge Test2NL Datasets (`merge_test2nl.py`)

Merges two Test2NL CSV datasets into a single combined dataset with contiguous IDs. Useful for combining results from multiple Test2NL generation runs (e.g., after recovering missing entries).

```
Input:  resources/test2nl/old_filtered_dataset/test2nl.csv
        resources/test2nl/missing_dataset/test2nl.csv
Output: resources/test2nl/filtered_dataset/test2nl.csv
```

```bash
python -m sakura.dataset_creation.merge_test2nl
```

The script handles duplicate entries by matching on `(project_name, qualified_class_name, method_signature, abstraction_level)`:
- If an entry from the second dataset matches one in the first, it replaces the first's entry but preserves the original ID
- Unique entries from the second dataset are appended with contiguous IDs starting after the first dataset's max ID

### 8. Verify Test2NL Dataset (`verify_test2nl.py`)

Validates the integrity of a Test2NL CSV dataset by checking:
1. IDs are contiguous starting from 0 with no gaps
2. All methods from the bucketed dataset are present with all three abstraction levels (low, medium, high)

```
Input:  resources/test2nl/filtered_dataset/test2nl.csv
        resources/filtered_bucketed_tests/<project>/nl2test.json
Output: Console output with validation results
```

```bash
python -m sakura.dataset_creation.verify_test2nl
```

### 9. Create Random Sample (Optional) (`create_random_sample_dataset.py`)

Generates random samples from each bucket for evaluation or debugging purposes.

```
Input:  resources/bucketed_tests/<project>/nl2test.json
Output: resources/sampled_tests/<project>/nl2test.json
```

```bash
python -m sakura.dataset_creation.create_random_sample_dataset
```

Default sample size: 20 per bucket

## Data Models

Defined in `model.py`:

- **Test**: Represents a single test method with qualified class name, method signature, and focal method details
- **NL2TestDataset**: Contains tests organized into buckets by focal method count

## Directory Structure

```
resources/
  datasets/              # Source Java project repositories (git submodules)
  analysis/              # CLDK analysis cache (auto-generated)
  hamster/               # Hamster model outputs
  hamster_models/        # Alternative hamster model location
  bucketed_tests/        # Tests bucketed by focal method count
  filtered_tests/        # Tests filtered by commit date
  filtered_bucketed_tests/  # Combined filtered + bucketed
  erroneous_tests/       # Tests that failed verification
  missing_tests/         # Tests missing from Test2NL output
  sampled_tests/         # Random samples for evaluation
  test2nl/               # Test2NL generated descriptions
    filtered_dataset/    # Current merged Test2NL output
    old_filtered_dataset/  # Previous Test2NL output (for merging)
    missing_dataset/     # Recovered missing Test2NL entries
```