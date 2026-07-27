package org.apache.bcel;

import org.apache.bcel.classfile.JavaClass;
import org.apache.bcel.generic.ClassGen;
import org.apache.bcel.generic.Type;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class RepositoryTest {

    @Test
    public void testFindFieldOnCircularInheritance() {
        ClassGen firstClassGen = new ClassGen("com.example.FirstClass", "java.lang.Object", "FirstClass.java", Const.ACC_PUBLIC, new String[]{"com.example.SecondInterface"});
        ClassGen secondInterfaceGen = new ClassGen("com.example.SecondInterface", "com.example.FirstClass", "SecondInterface.java", Const.ACC_PUBLIC | Const.ACC_INTERFACE, new String[]{});
        ClassGen thirdTestClassGen = new ClassGen("com.example.TestClass", "com.example.FirstClass", "TestClass.java", Const.ACC_PUBLIC, new String[]{});

        JavaClass firstClass = firstClassGen.getJavaClass();
        JavaClass secondInterface = secondInterfaceGen.getJavaClass();
        JavaClass thirdTestClass = thirdTestClassGen.getJavaClass();

        Repository.addClass(firstClass);
        Repository.addClass(secondInterface);
        Repository.addClass(thirdTestClass);

        try {
            assertThrows(ClassCircularityError.class, () -> {
                thirdTestClass.findField("nonExistentField", Type.INT);
            });
        } finally {
            Repository.removeClass(firstClass);
            Repository.removeClass(secondInterface);
            Repository.removeClass(thirdTestClass);
        }
    }
}
