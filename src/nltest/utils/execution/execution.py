import os
import subprocess
from typing import Optional, Dict

from nltest.utils.constants import MAVEN_CMD


class JavaExecution:
    """
    Execute tests in a Maven-based Java project.
    """
    @staticmethod
    def execute(project_root: str, test_class_name: Optional[str] = None, timeout: int = 900) -> Dict[str, object]:
        pom = os.path.join(project_root, "pom.xml")

        # Build the Maven command
        cmd = [MAVEN_CMD, "-f", pom]
        cmd += ["-DfailIfNoTests=false"]

        if test_class_name:
            # Specify which test class to run
            cmd += [f"-Dtest={test_class_name}", "test"]
        else:
            # Run full test phase
            cmd += ["test"]

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
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