package tests.inputs.packageTests
imports  tests.inputs.packageTestsDependencies.sig_A_1
       , tests.inputs.packageTestsDependencies.sig_A_2;

signature B = {
    use A; // should fail
    use tests.inputs.packageTestsDependencies.sig_A_2.A; // should succeed
}
