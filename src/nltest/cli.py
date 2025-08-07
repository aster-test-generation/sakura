from pathlib import Path

import typer
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from typing_extensions import Annotated

from nltest.test2nl.generation import DescriptionGenerator
from nltest.test2nl.model.models import AbstractionLevel
from nltest.test2nl import Pipeline
from nltest.utils import Config
from nltest.utils import constants
from nltest.utils.constants import LLM_CONFIG
from nltest.utils.pretty.color_logger import RichLog

app = typer.Typer(
    help="ASTER-Test2NL: [A]utomated Te[s][t] Cas[e] Generato[r] from Natural Language",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    add_completion=False
)


@app.callback()
def main() -> None:
    """Handles the global options"""
    return


@app.command()
def generate_descriptions(
        config_file: Annotated[
            str,
            typer.Option(
                help="Path to the configuration TOML file.",
                show_default=False
            ),
        ] = "./configs/vela.toml",
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
        description_temp: Annotated[
            float,
            typer.Option(
                help="Temperature used for the LLM-generated natural language description of the test case.",
                show_default=False,
            )
        ] = 0.5,
        code_gen_temp: Annotated[
            float,
            typer.Option(
                help="Temperature used for the LLM-generated code from the natural language description of the test case.",
                show_default=False,
            )
        ] = 0.1,
        evaluate: Annotated[
            bool,
            typer.Option(
                help="Whether to evaluation the description for each test case.",
                show_default=False,
            )
        ] = False,
):
    project_root = Path(project_root)
    if not (project_root.exists() and project_root.is_dir()):
        raise Exception(f"Project root directory {project_root} does not exist.")
    project_name = project_root.name

    config_file = Path(config_file)
    if not (config_file.exists() and config_file.is_file()):
        raise Exception(f"Configuration file {config_file} does not exist.")
    config = Config(config_file, reuse=False)

    if not output_dir:
        output_dir = f"./output/{project_name}"
    output_dir = Path(output_dir)

    # Select LLM
    # TODO: Change later
    llm_model = "DEEPSEEK-R1"

    # Assign VELA model configs
    llm_provider = config.get("llm_provider", "name")
    config.set(llm_provider, "model_id", val=LLM_CONFIG[llm_model]["identifier"])
    config.set(llm_provider, "API_URL", val=LLM_CONFIG[llm_model]["api_url"])
    config.set(llm_provider, "desc_temp", description_temp)
    config.set(llm_provider, "code_gen_temp", code_gen_temp)

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

    # Create orchestration orchestration to handle workflow
    pipeline = Pipeline(analysis, project_root, output_dir)
    for abs_level in AbstractionLevel:
        if evaluate:
            pipeline.run_all(abs_level, regen_classes=True)
        else:
            pipeline.run_descriptions(abs_level)


@app.command()
def generate(
        project_root: Annotated[
            str,
            typer.Option(
                help="Path to the root directory of the application under test.",
                show_default=False,
            ),
        ] = None,
        config_file: Annotated[
            str,
            typer.Option(help="Path to the configuration TOML file.", show_default=False),
        ] = "./configs/vela.toml",
        model_name: Annotated[
            str,
            typer.Option(
                help="Name of model to use for test generation "
            ),
        ] = "ibm-granite/granite-3.1-8b-instruct",
        source_root: Annotated[
            str,
            typer.Option(
                help="Root directory for application source (relative to project_root) "
                     "where the package structure starts."
            ),
        ] = "src/main/java",
        test_root: Annotated[
            str,
            typer.Option(
                help="Root directory for test cases (relative to project_root) where the package structure starts."
            ),
        ] = "src/test/java",

        nl2test_work_dir: Annotated[
            str,
            typer.Option(
                "--nl2test-work-dir",
                help="Path to directory where test generation log files are written; "
                     "default is the current working directory",
            ),
        ] = constants.DEBUG_DIR,
        report_coverage: Annotated[
            bool,
            typer.Option(
                "--report-coverage",
                help="If set, coverage will be reported at the end of the generation."
            ),
        ] = False,

        enable_test_saving: Annotated[
            bool,
            typer.Option(
                "--enable-test-saving",
                help="If set, generated tests will be saved to respected test folder."
            ),
        ] = False,
        analysis_json_path: Annotated[
            str,
            typer.Option(
                "--analysis-json-path",
                help="If set, analysis json file will be stored."
            ),
        ] = '',

) -> None:
    """Generate integration tests."""
    # read configuration from specified config file
    config = Config(config_file, reuse=False)

    # Update the config with the CLI args
    config.set("generation", "project_root", val=project_root)
    config.set("generation", "source_root", val=source_root)
    config.set("generation", "test_root", val=test_root)
    config.set(config.get("llm_provider", "name"), "model_id", val=model_name)
    config.set("generation", "nl2test_work_dir", val=nl2test_work_dir)
    config.set("generation", "report_coverage", val=report_coverage)
    config.set("generation", "enable_test_saving", val=enable_test_saving)

    config.set("generation", "analysis_json_path", val=analysis_json_path)
    DescriptionGenerator().generate_for_project()


if __name__ == "__main__":
    app()
