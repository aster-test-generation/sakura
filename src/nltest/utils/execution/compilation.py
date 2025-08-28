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

        # Ensure formatting
        subprocess.check_call([
            MAVEN_CMD,
            "-f", pom,
            "spring-javaformat:apply"
        ])

        p = subprocess.Popen(
            [
                MAVEN_CMD,
                "-f", pom,
                "-Drat.skip=true",
                "-Dstyle.color=never",
                "clean",
                "compile",
                "compiler:testCompile"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Call the sub-processs
        out, _ = p.communicate()

        for line in out.decode("utf-8").splitlines():
            if "[ERROR]" in line and ".java" in line:
                error_classes.add(line.split(':[')[0].split('/')[-1])

        return list(error_classes)