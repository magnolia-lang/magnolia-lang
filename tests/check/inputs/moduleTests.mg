package tests.check.inputs.moduleTests;

implementation ASourceExternal = external C++ path.to.file.dot.struct {
    type T;
    function id(t: T): T;
}

// module reference casting tests

// should succeed
signature DerivedSignatureWithCast = signature(ASourceExternal);

// should succeed
concept DerivedConceptWithCast = signature(ASourceExternal);

// should succeed
implementation DerivedImplementationWithCast = signature(ASourceExternal);

// should fail
program DerivedProgramWithCast = signature(ASourceExternal);

// should succeed
implementation DerivedExternalWithCast =
    external C++ path.to.file.dot.struct2 signature(ASourceExternal);

// should fail
implementation DerivedExternalWithoutCastFromExternal =
    external C++ path.to.file.dot.struct3 ASourceExternal;

// should succeed
program DerivedExternalWithCastAsProgram = DerivedExternalWithCast;

// not a test -- should succeed
implementation ASourceImplementation = {
    type T;
    function id(t: T): T = t;
}

// should fail
implementation DerivedExternalWithoutCastFromProgram =
    external C++ path.to.file.dot.struct4 ASourceImplementation;

// should succeed
program UseASourceExternal = {
    use ASourceExternal;
}

// should fail
program UseASourceExternalAsSignature = {
    use signature(ASourceExternal);
}
