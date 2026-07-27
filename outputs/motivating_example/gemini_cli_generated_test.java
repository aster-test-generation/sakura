package org.apache.commons.bcel;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.apache.commons.bcel.classfile.ClassFormatException;
import org.apache.commons.bcel.classfile.JavaClass;
import org.apache.commons.bcel.generic.ClassGen;
import org.apache.commons.bcel.generic.ConstantPoolGen;
import org.junit.jupiter.api.Test;

public class CircularInheritanceTest {

    @Test
    public void testCircularInheritanceDetection() throws ClassNotFoundException {
        final String circularAClassName = "org.apache.commons.bcel.test.CircularA";
        final String circularBInterfaceName = "org.apache.commons.bcel.test.CircularB";
        final String testClassName = "org.apache.commons.bcel.test.TestClass";

        // 1. Prepare interconnected type definitions.
        // Create Class A: a class that extends an interface (Interface B).
        // This is not valid in Java, but we are testing BCEL's ability to handle it.
        final ConstantPoolGen cpA = new ConstantPoolGen();
        final ClassGen classA = new ClassGen(circularAClassName, circularBInterfaceName, "CircularA.java", Const.ACC_PUBLIC,
            new String[0], cpA);
        final JavaClass javaClassA = classA.getJavaClass();

        // Create Interface B: an interface that extends a class (Class A).
        // This forms the circular dependency.
        final ConstantPoolGen cpB = new ConstantPoolGen();
        final ClassGen interfaceB = new ClassGen(circularBInterfaceName, "java.lang.Object", "CircularB.java",
            Const.ACC_PUBLIC | Const.ACC_INTERFACE, new String[] { circularAClassName }, cpB);
        final JavaClass javaClassB = interfaceB.getJavaClass();

        // Create TestClass: a class that extends Class A.
        final ConstantPoolGen cpC = new ConstantPoolGen();
        final ClassGen testClassGen = new ClassGen(testClassName, circularAClassName, "TestClass.java", Const.ACC_PUBLIC,
            new String[0], cpC);
        final JavaClass javaTestClass = testClassGen.getJavaClass();

        try {
            // 2. Register definitions in the repository.
            Repository.addClass(javaClassA);
            Repository.addClass(javaClassB);
            Repository.addClass(javaTestClass);

            // 3. Look up the test class.
            final JavaClass lookupTestClass = Repository.lookupClass(testClassName);

            // 4. Assert that searching for a field triggers a circular dependency error.
            // getFields() traverses the class hierarchy, which should detect the cycle.
            assertThatThrownBy(() -> lookupTestClass.getFields())
                .isInstanceOf(ClassFormatException.class)
                .hasMessageContaining("Circular inheritance reference");

        } finally {
            // 5. Clean up the repository to restore its original state.
            Repository.removeClass(javaClassA);
            Repository.removeClass(javaClassB);
            Repository.removeClass(javaTestClass);
        }
    }
}
