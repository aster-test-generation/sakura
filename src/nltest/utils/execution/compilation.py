import os
import subprocess
from typing import List

from nltest.utils.constants import MAVEN_CMD


class JavaCompilation:
    """
    Compile and prepare Maven-based Java project for testing.
    """

    @staticmethod
    def get_erroneous_classes(project_root) -> List[str]:
        error_classes = set()
        pom = os.path.join(project_root, "pom.xml")

        def _run_compile(pom_path):
            return subprocess.run(
                [
                    MAVEN_CMD,
                    "-f",
                    pom_path,
                    "-Drat.skip=true",
                    "-Dstyle.color=never",
                    "clean",
                    "compile",
                    "compiler:testCompile",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        def _should_try_formatting(stdout: str, stderr: str) -> bool:
            text = f"{stdout or ''}\n{stderr or ''}".lower()
            triggers = (
                "spring-javaformat",
                "spring-javaformat:check",
                "spring javaformat",
                "format violation",
                "please run 'mvn spring-javaformat:apply'",
            )
            return any(t in text for t in triggers)

        def _apply_format(pom_path) -> bool:
            try:
                subprocess.check_call([MAVEN_CMD, "-f", pom_path, "spring-javaformat:apply"])  # type: ignore[arg-type]
                return True
            except Exception:
                return False

        # Run compile first
        result = _run_compile(pom)

        # Format if compile fails and formatting required
        if result.returncode != 0 and _should_try_formatting(
            result.stdout, result.stderr
        ):
            if _apply_format(pom):
                result = _run_compile(pom)

        # Parse compile attempt
        out_text = result.stdout or ""
        for line in out_text.splitlines():
            if "[ERROR]" in line and ".java" in line:
                error_classes.add(line.split(":[")[0].split("/")[-1])

        return list(error_classes)
