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
python -m nltest.dataset_creation.create_hamster_model
```

### 2. Bucketize Dataset (`bucketize_dataset.py`)

Processes hamster models to categorize tests by focal method count into buckets (1, 2, 3-5, 6-10, >10 focal methods). Filters out abstract classes, interfaces, enums, and annotation declarations.

```
Input:  resources/hamster_models/<project>/hamster.json
Output: resources/bucketed_tests/<project>/nl2test.json
```

```bash
python -m nltest.dataset_creation.bucketize_dataset
```

### 3. Filter by Date (`filter_by_date.py`)

Identifies test methods added to repositories after a specified date using git history and Tree-sitter parsing. Can run independently of steps 1-2.

```
Input:  resources/datasets/<project>/        (Git repositories)
Output: resources/filtered_tests/<project>/nl2test.json
        resources/filtered_tests/summary.json
```

```bash
python -m nltest.dataset_creation.filter_by_date
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
python -m nltest.dataset_creation.bucketize_filtered_dataset
```

### 5. Create Random Sample (Optional) (`create_random_sample_dataset.py`)

Generates random samples from each bucket for evaluation or debugging purposes.

```
Input:  resources/bucketed_tests/<project>/nl2test.json
Output: resources/sampled_tests/<project>/nl2test.json
```

```bash
python -m nltest.dataset_creation.create_random_sample_dataset
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
  sampled_tests/         # Random samples for evaluation
```