import glob
import os
from pathlib import Path

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.utils import Config
from nltest.utils import constants
from nltest.utils.pretty import RichLog


class NL2TestHelper:
    def __init__(self):
        self.config = Config()
        llm_provider = self.config.get("llm_provider", "name")
        self.target_class = self.config.get("generation", "target_class")
        self.target_method = self.config.get("generation", "target_method")
        self.project_root = self.config.get("generation", "project_root")
        self.user_provided_analysis_json_base_path = self.config.get("generation", "analysis_json_path")
        self.dataset_path = Path(self.project_root).parent
        self.project_name = Path(self.project_root).name
        self.model_id = self.config.get(llm_provider, "model_id")
        self.source_root = self.config.get("generation", "source_root")
        self.enable_test_saving = self.config.get("generation", "enable_test_saving")
        self.test_root = self.config.get("generation", "test_root")
        self.app_source_path = Path.cwd().joinpath(self.dataset_path, self.project_name, self.source_root).__str__()
        self.root_package = self._get_root_package()
        if self.user_provided_analysis_json_base_path == '':
            self.analysis_path = Path.cwd().joinpath(constants.DEFAULT_ANALYSIS_DIR, self.project_name)
        else:
            self.analysis_path = Path(self.user_provided_analysis_json_base_path).joinpath(
                constants.DEFAULT_ANALYSIS_DIR, self.project_name)
        self.save_path = Path.cwd().joinpath(
            self.dataset_path,
            self.project_name,
            self.test_root,
            self.root_package.replace(".", os.sep)
        )

        # create CLDK object, which computes program analysis information
        RichLog.info(f"Gathering static analysis results")
        project_dir = Path.cwd().joinpath(self.dataset_path, self.project_name).__str__()
        self.analysis = CLDK(language="java").analysis(
            project_path=project_dir,
            # analysis_backend="codeanalyzer",
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.analysis_path,
        )
        RichLog.info(f"Successfully finished gathering static analysis results")

    def _get_root_package(self) -> str:
        """
        Computes and returns root package name for the application under test.

        Computes root package by finding the common path prefix of all directories under application
        source root that contain java source files.

        Returns:
            str: root package for app
        """
        app_dirs = []
        # ignored_java_files = ["package-info.java", "module-info.java"]
        for path, subdirs, files in os.walk(self.app_source_path):
            for d in subdirs:
                java_src_files = glob.glob(os.path.join(path, d, "*.java"), recursive=False)
                if java_src_files:
                    app_dirs.append(os.path.join(path, d))
        common_path_prefix = os.path.commonprefix(app_dirs)
        root_package = (common_path_prefix.removeprefix(self.app_source_path).strip(os.path.sep).replace(
            os.path.sep, "."))
        return root_package

