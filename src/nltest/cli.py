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
from nltest.nl2test.model.models import NL2TestInput
from nltest.test2nl.model.models import Test2NLEntry
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
        project_root: Annotated[
            str,
            typer.Option(
                help="Path to the root directory of the application under test.",
                show_default=False,
            ),
        ] = "./resources/spring-petclinic",
        output_dir: Annotated[
            str,
            typer.Option(
                help="Path to the output directory for saving descriptions.",
                show_default=False,
            ),
        ] = None,
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
):
    # Select LLM
    # TODO: Change later
    llm_model = "DEEPSEEK-R1"

    project_root = Path(project_root)
    if not (project_root.exists() and project_root.is_dir()):
        raise Exception(f"Project root directory {project_root} does not exist.")
    project_name = project_root.name

    if not output_dir:
        output_dir = f"./output/{project_name}"
    output_dir = Path(output_dir)

    config = init_config(
        project_name=project_name,
        base_project_dir=str(project_root),
        output_dir=str(output_dir),
        llm_provider=Provider.OPENROUTER,
        llm_model=llm_model,
        llm_api_key=os.getenv("OPENROUTER_API_KEY"),  # Assign from env
    )

    # Generate analysis of the current project
    RichLog.info(f"Gathering static analysis results")
    analysis = CLDK(language="java").analysis(
        project_path=project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=output_dir,
        eager=True,
    )
    RichLog.info(f"Successfully finished gathering static analysis results")

    # Create orchestration to handle workflow
    pipeline = Pipeline(analysis, project_root, output_dir)

    if clear_dataset:
        RichLog.info("Clearing the existing dataset at the output directory.")
        pipeline.reset_dataset()

    for abs_level in AbstractionLevel:
        if evaluate:
            pipeline.run_all(abs_level, regen_classes=True)
        else:
            pipeline.run_descriptions(abs_level)


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
        ] = "moonshotai/kimi-k2",
        emb_model: Annotated[
            str,
            typer.Option(
                help="Embedding model to use for vector search.",
                show_default=False,
            ),
        ] = "nomic-embed-text:v1.5",
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
                help="Maximum number of entries to process (0 for all).",
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
    RichLog.info(f"Loaded {len(test2nl_entries)} Test2NL entries")

    # Limit entries if specified
    if max_entries > 0:
        test2nl_entries = test2nl_entries[:max_entries]
        RichLog.info(f"Processing first {len(test2nl_entries)} entries")

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

    # Process each project separately
    total_successful_evaluations = 0
    total_failed_evaluations = 0
    
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
        pipeline = NL2TestPipeline(analysis, project_root)

        # Process inputs for this project
        total_inputs = len(nl2test_inputs)
        successful_evaluations = 0
        failed_evaluations = 0
        
        RichLog.info(f"Starting localization evaluation for {total_inputs} inputs in {project_name}")
        
        for i, nl2test_input in enumerate(nl2test_inputs, 1):
            try:
                RichLog.info(f"Processing input {i}/{total_inputs}: {nl2test_input.qualified_class_name}.{nl2test_input.method_signature}")
                
                # Run localization evaluation pipeline
                localized_blocks, coverage_score = pipeline.run_localization_evaluation_pipeline(nl2test_input)
                
                RichLog.info(f"Coverage score: {coverage_score:.3f}")
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
    
    if save_results:
        RichLog.info(f"Detailed results saved to {output_dir}")


if __name__ == "__main__":
    app()
