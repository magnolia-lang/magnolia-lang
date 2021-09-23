package tests.check.inputs.requirementTests;

signature Req = {
    type T;
    function f(): T;
}

implementation IExtReq1 = external C++ File.ExtReq1 {
    type T;
    function f(): T;
}

implementation IExtReq2 = external C++ File.ExtReq2 {
    type T;
    function f(): T;
}

implementation IExtReq3 = external C++ File.ExtReq3 {
    type T;
    require type T; // should succeed
    type T; // should succeed; this was previously tagged as "should fail", but
            // within the module expression, this is a perfectly valid
            // declaration; the externalization of the declarations should
            // happen after the whole module expression has been reduced, i.e.
            // this whole module expression reduces to "type T;".
}

implementation IExtReq4 = external C++ File.ExtReq4 {
    type T;
    require function f(): T;
}

implementation MergingImpls = {
    use IExtReq1;
    use IExtReq2; // should fail
    use IExtReq1; // should succeed
    require type T; // should succeed
    type T; // should succeed
    function f(): T; // should succeed
    function f(): T; // should succeed
    require function f(): T; // should succeed
}

implementation MergingGuards = {
    type T;
    predicate g_1();
    predicate g_2();

    function f(): T guard g_1(); // should succeed
    function f(): T guard g_2(); // should succeed
}

// should fail
program MissingRequiredFunction = {
    use IExtReq4;
}

// should fail
implementation ImplementingARequirement = {
    type T;
    function impl(): T;
    require function f(): T = impl();
}

// should fail
program PExtReq1 = external C++ File.ExtReq5 {
    require IExtReq3;
}

// should succeed
program PExtReq2 = external C++ File.ExtReq6 {
    use signature(IExtReq3);
}
