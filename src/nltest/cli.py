import os
from pathlib import Path
from dotenv import load_dotenv

import typer
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from typing_extensions import Annotated

from nltest.test2nl.generation import DescriptionGenerator
from nltest.test2nl.model.models import AbstractionLevel
from nltest.test2nl import Pipeline
from nltest.utils.config import Config, init_config
from nltest.utils import constants
from nltest.utils.llm.model import Provider
from nltest.utils.pretty.color_logger import RichLog
from nltest.utils.pretty.prints import pretty_print
from nltest.nl2test import Pipeline as NL2TestPipeline
from nltest.nl2test.models import NL2TestInput, NL2LocalizationOutput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.test2nl.model.models import Test2NLEntry, TestDescriptionInfo
from nltest.utils.file_io.structured_data_manager import StructuredDataManager

app = typer.Typer(
    help="ASTER-Test2NL: [A]utomated Te[s][t] Cas[e] Generato[r] from Natural Language",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    add_completion=False
)

load_dotenv()

@app.callback()
def main() -> None:
    """Handles the global options"""
    return


@app.command()
def generate_descriptions(
        base_project_dir: Annotated[
            str,
            typer.Option(
                help="Path to the base directory containing the projects.",
                show_default=False,
            ),
        ] = "./resources",
        output_dir: Annotated[
            str,
            typer.Option(
                help="Path to the output directory for saving descriptions.",
                show_default=False,
            ),
        ] = None,
        llm_model: Annotated[
            str,
            typer.Option(
                help="LLM model to use for description generation.",
                show_default=False,
            ),
        ] = "mistralai/devstral-small",
        evaluate: Annotated[
            bool,
            typer.Option(
                help="Whether to evaluation the description for each test case.",
                show_default=False,
            )
        ] = False,
        clear_dataset: Annotated[
            bool,
            typer.Option(
                help="Whether to delete the existing descriptions at the save location, or whether to append to them.",
                show_default=False,
            )
        ] = True,
        max_entries: Annotated[
            int,
            typer.Option(
                help="Maximum number of Test2NL entries to generate per project (0 for unlimited).",
                show_default=False,
            )
        ] = 0,
        only_interesting_tests: Annotated[
            bool,
            typer.Option(
                help="Whether to only generate descriptions for interesting/complicated tests (those with complex focal class/method relationships).",
                show_default=False,
            )
        ] = False,
):
    base_project_dir = Path(base_project_dir)
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    if not output_dir:
        output_dir = f"./output"
    output_dir = Path(output_dir)

    # Get all project directories in base_project_dir
    all_projects = sorted([p for p in base_project_dir.iterdir() if p.is_dir()])
    
    if not all_projects:
        RichLog.error(f"No project directories found in {base_project_dir}")
        return

    RichLog.info(f"Found {len(all_projects)} project(s) to process: {[p.name for p in all_projects]}")

    # Process each project separately
    for project_root in all_projects:
        project_name = project_root.name
        RichLog.info(f"\n{'='*60}")
        RichLog.info(f"Processing project: {project_name}")
        RichLog.info(f"{'='*60}")

        # Create project-specific output directory
        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)

        config = init_config(
            project_name=project_name,
            base_project_dir=str(project_root),
            output_dir=str(project_output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            llm_api_key=os.getenv("OPENROUTER_API_KEY"),  # Assign from env
        )

        # Generate analysis of the current project
        RichLog.info(f"Gathering static analysis results for {project_name}")
        analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=project_output_dir,
            eager=True,
        )
        RichLog.info(f"Successfully finished gathering static analysis results for {project_name}")

        # Create orchestration to handle workflow
        pipeline = Pipeline(analysis, project_root, project_output_dir)

        if clear_dataset:
            RichLog.info(f"Clearing the existing dataset for {project_name} at the output directory.")
            pipeline.reset_dataset()

        # Track total entries generated for this project
        total_entries_generated = 0
        
        for abs_level in AbstractionLevel:
            # Calculate remaining entries for this abstraction level
            remaining_entries = max_entries - total_entries_generated if max_entries > 0 else 0
            
            # Break early if we've reached the limit
            if max_entries > 0 and total_entries_generated >= max_entries:
                RichLog.info(f"Reached maximum entries limit ({max_entries}) for {project_name}. Stopping early.")
                break
                
            if evaluate:
                pipeline.run_all(abs_level, regen_classes=True, max_entries=remaining_entries, only_interesting_tests=only_interesting_tests)
                # Get the number of descriptions generated (we need to load them to count)
                try:
                    descriptions = pipeline.data_manager.load("descriptions.json", TestDescriptionInfo)
                    # Count descriptions for this abstraction level and project
                    entries_generated = len([d for d in descriptions if d.abstraction_level == abs_level])
                    total_entries_generated += entries_generated
                    
                    RichLog.info(f"Generated {entries_generated} entries for {abs_level.value} abstraction level")
                    RichLog.info(f"Total entries generated for {project_name}: {total_entries_generated}")
                    
                    # Break early if we've reached the limit
                    if max_entries > 0 and total_entries_generated >= max_entries:
                        RichLog.info(f"Reached maximum entries limit ({max_entries}) for {project_name}.")
                        break
                except FileNotFoundError:
                    pass
            else:
                # Pass the remaining entries limit to run_descriptions
                descriptions = pipeline.run_descriptions(abs_level, max_entries=remaining_entries, only_interesting_tests=only_interesting_tests)
                entries_generated = len(descriptions)
                total_entries_generated += entries_generated
                
                RichLog.info(f"Generated {entries_generated} entries for {abs_level.value} abstraction level")
                RichLog.info(f"Total entries generated for {project_name}: {total_entries_generated}")
                
                # Break early if we've reached the limit
                if max_entries > 0 and total_entries_generated >= max_entries:
                    RichLog.info(f"Reached maximum entries limit ({max_entries}) for {project_name}.")
                    break

        final_message = f"Completed processing project: {project_name}"
        if max_entries > 0:
            final_message += f" (Generated {total_entries_generated}/{max_entries} entries)"
        RichLog.info(final_message)

    RichLog.info(f"\n{'='*60}")
    RichLog.info(f"Completed description generation for all {len(all_projects)} projects")
    RichLog.info(f"{'='*60}")


@app.command()
def evaluate_localization(
        base_project_dir: Annotated[
            str,
            typer.Option(
                help="Path to the base directory containing the projects.",
                show_default=False,
            ),
        ] = "./resources",
        output_dir: Annotated[
            str,
            typer.Option(
                help="Path to the output directory containing the test2nl.csv file.",
                show_default=False,
            ),
        ] = None,
        csv_file: Annotated[
            str,
            typer.Option(
                help="Name of the CSV file containing Test2NL entries (default: test2nl.csv).",
                show_default=False,
            ),
        ] = "test2nl.csv",
        llm_model: Annotated[
            str,
            typer.Option(
                help="LLM model to use for localization evaluation.",
                show_default=False,
            ),
        ] = "mistralai/devstral-small",
        emb_model: Annotated[
            str,
            typer.Option(
                help="Embedding model to use for vector search.",
                show_default=False,
            ),
        ] = "nomic-embed-text:v1.5",
        decomposition_mode: Annotated[
            str,
            typer.Option(
                help="Decomposition mode: grammatical or gherkin",
                show_default=True,
            ),
        ] = "grammatical",
        save_results: Annotated[
            bool,
            typer.Option(
                help="Whether to save detailed evaluation results to files.",
                show_default=False,
            )
        ] = True,
        max_entries: Annotated[
            int,
            typer.Option(
                help="Maximum number of entries to process (0 for all). Entries are sorted by class-method pairs.",
                show_default=False,
            )
        ] = 0,
):
    base_project_dir = Path(base_project_dir)
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    if not output_dir:
        output_dir = f"./output"
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise Exception(f"Output directory {output_dir} does not exist.")

    # Load Test2NL entries from CSV
    data_manager = StructuredDataManager(output_dir)
    csv_path = output_dir / csv_file
    
    if not csv_path.exists():
        raise Exception(f"CSV file {csv_path} does not exist.")
    
    RichLog.info(f"Loading Test2NL entries from {csv_path}")
    test2nl_entries = data_manager.load(csv_file, Test2NLEntry, format="csv")
    
    # Sort entries by qualified_class_name and method_signature to group related entries together
    test2nl_entries.sort(key=lambda entry: (entry.qualified_class_name, entry.method_signature))
    RichLog.info(f"Loaded and sorted {len(test2nl_entries)} Test2NL entries by class-method pairs")

    # Limit entries if specified (applied to individual entries, not class-method pairs)
    if max_entries > 0:
        test2nl_entries = test2nl_entries[:max_entries]
        RichLog.info(f"Processing first {len(test2nl_entries)} entries (max_entries={max_entries})")

    # Convert Test2NL entries to NL2TestInput objects and organize by project
    nl2test_inputs_by_project = {}
    for entry in test2nl_entries:
        nl2test_input = NL2TestInput(
            description=entry.description,
            project_name=entry.project_name,
            qualified_class_name=entry.qualified_class_name,
            method_signature=entry.method_signature,
            abstraction_level=entry.abstraction_level.value,
            is_bdd=entry.is_bdd,
            id=entry.id
        )
        
        if entry.project_name not in nl2test_inputs_by_project:
            nl2test_inputs_by_project[entry.project_name] = []
        nl2test_inputs_by_project[entry.project_name].append(nl2test_input)

    RichLog.info(f"Organized inputs by {len(nl2test_inputs_by_project)} projects: {list(nl2test_inputs_by_project.keys())}")

    # Normalize decomposition mode
    try:
        mode_enum = DecompositionMode(decomposition_mode)
    except ValueError:
        raise typer.BadParameter("decomposition_mode must be 'grammatical' or 'gherkin'")

    # Process each project separately
    total_successful_evaluations = 0
    total_failed_evaluations = 0
    all_localization_outputs = []
    
    for project_name, nl2test_inputs in nl2test_inputs_by_project.items():
        RichLog.info(f"\n{'='*60}")
        RichLog.info(f"Processing project: {project_name}")
        RichLog.info(f"{'='*60}")
        
        # Create project-specific paths
        project_root = base_project_dir / project_name
        if not (project_root.exists() and project_root.is_dir()):
            RichLog.error(f"Project directory {project_root} does not exist. Skipping project {project_name}.")
            continue
            
        project_output_dir = output_dir / project_name
        
        # Initialize configuration for this project
        config = init_config(
            project_name=project_name,
            base_project_dir=str(project_root),
            output_dir=str(project_output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            emb_provider=Provider.OLLAMA,
            emb_model=emb_model,
            llm_api_key=os.getenv("OPENROUTER_API_KEY"),
            emb_api_key=None,
        )

        # Generate analysis for this specific project
        RichLog.info(f"Gathering static analysis results for {project_name}")
        analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=project_output_dir,
            eager=True,
        )
        RichLog.info(f"Successfully finished gathering static analysis results for {project_name}")

        # Create NL2Test pipeline for this project
        RichLog.info(f"Initializing NL2Test pipeline for {project_name}")
        pipeline = NL2TestPipeline(analysis, project_root, decomposition_mode=mode_enum)

        # Process inputs for this project
        total_inputs = len(nl2test_inputs)
        successful_evaluations = 0
        failed_evaluations = 0
        
        RichLog.info(f"Starting localization evaluation for {total_inputs} inputs in {project_name}")
        
        for i, nl2test_input in enumerate(nl2test_inputs, 1):
            try:
                RichLog.info(f"Processing input {i}/{total_inputs}: {nl2test_input.qualified_class_name}.{nl2test_input.method_signature} ({nl2test_input.abstraction_level})")
                
                # Run localization evaluation pipeline
                localization_output = pipeline.run_localization_evaluation_pipeline(nl2test_input)
                all_localization_outputs.append(localization_output)
                
                pretty_print("Localized Blocks:", localization_output.localized_blocks)
                RichLog.info(f"Coverage score: {localization_output.coverage_score:.3f}")
                successful_evaluations += 1

            except Exception as e:
                RichLog.error(f"Failed to process input {i}: {str(e)}")
                failed_evaluations += 1
                continue

        RichLog.info(f"Project {project_name} evaluation completed:")
        RichLog.info(f"  - Total inputs: {total_inputs}")
        RichLog.info(f"  - Successful evaluations: {successful_evaluations}")
        RichLog.info(f"  - Failed evaluations: {failed_evaluations}")
        
        total_successful_evaluations += successful_evaluations
        total_failed_evaluations += failed_evaluations

    RichLog.info(f"\n{'='*60}")
    RichLog.info(f"Overall localization evaluation completed:")
    RichLog.info(f"  - Total projects processed: {len(nl2test_inputs_by_project)}")
    RichLog.info(f"  - Total successful evaluations: {total_successful_evaluations}")
    RichLog.info(f"  - Total failed evaluations: {total_failed_evaluations}")
    RichLog.info(f"{'='*60}")
    
    if save_results and all_localization_outputs:
        # Save all localization outputs to a single file
        global_data_manager = StructuredDataManager(output_dir)
        filename = "nl2_localization_outputs.json"
        global_data_manager.save(filename, all_localization_outputs, format="json")
        RichLog.info(f"Saved {len(all_localization_outputs)} localization outputs to {output_dir / filename}")
    elif save_results:
        RichLog.info(f"No successful evaluations to save")


if __name__ == "__main__":
    app()
