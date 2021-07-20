package tests.check.inputs.packageTests
imports  tests.check.inputs.packageTestsDependencies.sig_A_1
       , tests.check.inputs.packageTestsDependencies.sig_A_2;

signature B = {
    use A; // should fail
    use tests.check.inputs.packageTestsDependencies.sig_A_2.A; // should succeed
}
