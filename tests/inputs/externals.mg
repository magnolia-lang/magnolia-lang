package tests.inputs.externals;

// TODO: implement external handling, and add tests.
implementation ICxx = external C++ some.cxx.file {
    type T;
}

implementation IJS = external JavaScript some.js.file {
    type T;
}

implementation IPy = external Python some.py.file {
    type T;
}

// should succeed
implementation UnifySameConcreteType = {
    use ICxx;
    use ICxx;
}

// should fail
implementation UnifyDifferentConcreteType = {
    use ICxx;
    use IJS;
}

// should succeed
implementation UnifyRequiredAndConcreteType = {
    use ICxx;
    type T;
}
