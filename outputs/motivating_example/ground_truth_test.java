    @Test
    void testFindFieldCustomClass() throws Exception {
        final byte[] classABytes = createClass("CyclicClassA", "CyclicClassB");
        final byte[] classBBytes = createInterface("CyclicClassB", "CyclicClassA");
        final byte[] testClassBytes = createClass("CyclicTestClass", "CyclicClassA");
        final JavaClass interfaceA = new ClassParser(new ByteArrayInputStream(classABytes), "CyclicClassA.class").parse();
        final JavaClass interfaceB = new ClassParser(new ByteArrayInputStream(classBBytes), "CyclicClassB.class").parse();
        final JavaClass testClass = new ClassParser(new ByteArrayInputStream(testClassBytes), "CyclicTestClass.class").parse();
        final SyntheticRepository repo = SyntheticRepository.getInstance();
        try {
            repo.storeClass(interfaceA);
            repo.storeClass(interfaceB);
            repo.storeClass(testClass);
            Repository.setRepository(repo);
            assertThrows(ClassCircularityError.class, () -> testClass.findField("nonExistentField", Type.INT));
        } finally {
            repo.removeClass(interfaceA);
            repo.removeClass(interfaceB);
            repo.removeClass(testClass);
        }
    }
