TEST_CODE = """
package <package_name>;

import org.jacoco.agent.rt.RT;
import org.jacoco.agent.rt.IAgent;

import org.junit.jupiter.api.extension.*;

import java.io.File;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class JaCoCoTestWatcher implements AfterTestExecutionCallback, BeforeTestExecutionCallback {

	private static final String BASE_DIR = "target/jacoco-tests";

	@Override
	public void beforeTestExecution(ExtensionContext context) throws Exception {
		// Optional: reset coverage at start of each test
		IAgent agent = RT.getAgent();
		agent.reset();
	}

	@Override
	public void afterTestExecution(ExtensionContext context) throws Exception {
		String testName = context.getRequiredTestClass().getName() + "__" +
				context.getRequiredTestMethod().getName();

		Files.createDirectories(Paths.get(BASE_DIR));

		byte[] execData = RT.getAgent().getExecutionData(false); // false = don't reset after reading

		try (OutputStream out = Files.newOutputStream(Paths.get(BASE_DIR, testName + ".exec"))) {
			System.out.println(BASE_DIR + File.separator + testName + ".exec");
			out.write(execData);
		}
	}
}
"""
