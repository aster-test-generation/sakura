import os
import sys
import shutil
import subprocess
from typing import Optional, Dict, List

from nltest.utils.constants import MAVEN_CMD


class JavaExecution:
    """
    Execute tests in a Maven-based Java project.
    """

    @staticmethod
    def execute(
        project_root: str, test_class_name: Optional[str] = None, timeout: int = 900
    ) -> Dict[str, object]:
        pom = os.path.join(project_root, "pom.xml")

        def _resolve_maven_cmd_parts(project_root: str) -> List[str]:
            wrapper = "mvnw.cmd" if sys.platform == "win32" else "mvnw"
            wrapper_path = os.path.join(project_root, wrapper)
            if os.path.isfile(wrapper_path):
                if sys.platform != "win32" and not os.access(wrapper_path, os.X_OK):
                    return ["sh", wrapper_path]
                return [wrapper_path]
            mvn_path = shutil.which(MAVEN_CMD)
            if mvn_path:
                return [mvn_path]
            raise FileNotFoundError(
                f"Maven not found. Neither '{MAVEN_CMD}' on PATH nor wrapper '{wrapper}' at {project_root}."
            )

        # Build the Maven command
        cmd = _resolve_maven_cmd_parts(project_root) + ["-f", pom]
        cmd += ["-DfailIfNoTests=false"]

        if test_class_name:
            # Specify which test class to run
            cmd += [f"-Dtest={test_class_name}", "test"]
        else:
            # Run full test phase
            cmd += ["test"]

        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd),
            }
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            return {
                "returncode": -1,
                "stdout": out.decode("utf-8", errors="ignore"),
                "stderr": "Test execution timed out.",
                "command": " ".join(cmd),
            }

        return {
            "returncode": p.returncode,
            "stdout": out.decode("utf-8", errors="ignore"),
            "stderr": err.decode("utf-8", errors="ignore"),
            "command": " ".join(cmd),
        }
