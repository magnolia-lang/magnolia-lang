package tests.check.inputs.regressionTests;

// Referencing https://github.com/magnolia-lang/magnolia-lang/issues/35
implementation I35 = {

    type T;

    predicate some_pred();

    // should succeed
    procedure some_procedure() = {
        if some_pred() then some_pred() else some_pred();
    }
}