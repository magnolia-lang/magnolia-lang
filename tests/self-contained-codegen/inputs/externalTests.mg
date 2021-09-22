package tests.self-contained-codegen.inputs.externalTests;

// TODO: some parts of this file will have to be moved when targeting different
// backends

implementation MgImplType_JS = external JavaScript IncludeFile.TyStruct {
    type T;
}

implementation MgImplType_Cxx = external C++ IncludeFile.TyStruct {
    type T;
}

implementation MgImplCallable_JS = external JavaScript IncludeFile.FnStruct {
    require type T;
    function identity(t: T): T;
}

// should fail when generating non-JS code
program ProgramThatReliesOnWrongBackendType = {
    use MgImplType_JS;
}

// should fail
program ProgramThatReliesOnWrongBackendCallable = {
    use MgImplType_Cxx;
    use MgImplCallable_JS;
}

implementation CircularDependency1 = external C++ IncludeFile.circular_dep1 {
    require type A;
    type B;
}

implementation CircularDependency2 = external C++ IncludeFile.circular_dep2 {
    type A;
    require type B;
}

// should fail
program CircularDependency = {
    use CircularDependency1;
    use CircularDependency2;
}
