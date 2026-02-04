TEST_CODE = """
package org.apache.commons.jxpath;

import org.jacoco.agent.rt.IAgent;
import org.jacoco.agent.rt.RT;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

import java.io.File;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class JaCoCoTestRule implements TestRule {

    private static final String BASE_DIR = "target" + File.separator + "jacoco-tests";

    @Override
    public Statement apply(Statement base, Description description) {
        return new Statement() {
            @Override
            public void evaluate() throws Throwable {
                // before test execution, reset jacoco agent
                IAgent agent = RT.getAgent();
                agent.reset();
                try {
                    base.evaluate(); // run the test
                } finally {
                    // after test execution
                    String testName = description.getMethodName();
                    Files.createDirectories(Paths.get(BASE_DIR));

                    byte[] execData = RT.getAgent().getExecutionData(false); // false = don't reset after reading

                    try (OutputStream out = Files.newOutputStream(Paths.get(BASE_DIR, testName + ".exec"))) {
                        System.out.println(BASE_DIR + File.separator + testName + ".exec");
                        out.write(execData);
                    }
                }
            }
        };
    }
}
"""