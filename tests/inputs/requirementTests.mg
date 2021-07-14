package tests.inputs.requirementTests;

signature Req = {
    type T;
    function f(): T;
}

implementation IExtReq1 = external C++ ExtReq1 {
    type T;
    function f(): T;
}

implementation IExtReq2 = external C++ ExtReq2 {
    type T;
    function f(): T;
}

implementation IExtReq3 = external C++ ExtReq3 {
    type T;
    require type T; // should succeed
    type T; // should fail
}

implementation MergingImpls = {
    use IExtReq1;
    use IExtReq2; // should fail
    use IExtReq1; // should succeed
    require type T; // should succeed
    type T; // should succeed
    function f(): T; // should succeed
    function f(): T; // should succeed
}

implementation MergingGuards = {
    type T;
    predicate g_1();
    predicate g_2();

    function f(): T guard g_1(); // should succeed
    function f(): T guard g_2(); // should succeed
}
