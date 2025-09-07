import os
import sys
import shutil
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

        def _resolve_maven_cmd_parts(project_root: str) -> List[str]:
            """
            - If `<project_root>/mvnw[.cmd]` exists, use it (or `sh mvnw` if not executable on *nix).
            - Else, if `mvn` is on PATH, use that.
            - Else, raise FileNotFoundError with a helpful message.
            """
            wrapper = "mvnw.cmd" if sys.platform == "win32" else "mvnw"
            wrapper_path = os.path.join(project_root, wrapper)
            if os.path.isfile(wrapper_path):
                if sys.platform != "win32" and not os.access(wrapper_path, os.X_OK):
                    return ["sh", wrapper_path]
                return [wrapper_path]
            mvn_path = shutil.which(MAVEN_CMD)
            if mvn_path:
                return [mvn_path]
            # Not found anywhere
            raise FileNotFoundError(
                f"Maven not found. Neither '{MAVEN_CMD}' on PATH nor wrapper '{wrapper}' at {project_root}."
            )

        cmd_base = _resolve_maven_cmd_parts(project_root)

        def _run_compile(pom_path):
            return subprocess.run(
                cmd_base
                + [
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
                subprocess.check_call(
                    cmd_base + ["-f", pom_path, "spring-javaformat:apply"]
                )  # type: ignore[arg-type]
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
