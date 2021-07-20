package tests.check.inputs.stmtTests;

// TODO: expand with tests for all the statements/expressions

implementation StmtTests = {
    require type T1;
    require type T2;

    require function mkT1(): T1;
    require function mkT2(): T2;

    procedure tests() {
        var t1 = mkT1();
        // == assignment statement tests ==
        // Valid assignment
        t1 = mkT1();
        // Invalid assignment
        t1 = mkT2();
    }
}

// TODO: add more tests
implementation VarExprTests = {

    require type T;
    require type U;

    procedure conflicting_type_annotation() = {
        var x: T;
        // should succeed
        x: T;
        // should fail
        x: U;
    }
}

// TODO: add more tests
implementation CallExprTests = {

    require type T;
    require type U;

    function overload_return_type(): T;
    function overload_return_type(): U;

    procedure deduce_overloaded_type() = {

        // should succeed
        var explicit_T: T = overload_return_type();

        // should succeed
        var explicit_U: U = overload_return_type();

        // should fail
        var implicit_type = overload_return_type();
    }
}

implementation LetExprTests = {

    require type T;
    require type U;

    function constant_T(): T;

    procedure var_assignment() = {
        // should succeed
        var x: T;
        // should fail
        var x: U;

    }

    procedure var_declaration_infer_type() = {

        var x;
        // should fail
        x = constant_T();
    }
}

implementation IfExprTests = {

    require type T;
    require type U;

    predicate pred();
    function constant_T(): T;
    function constant_U(): U;

    // should succeed
    procedure assign_if_to_var() = {
        var x = if pred() then constant_T() else constant_T();
    }

    // should succeed
    function returns_value_if(): T = {
        value if pred() then constant_T() else constant_T();
    }

    // should fail
    procedure type_mismatch_branches() = {
        if pred() then constant_T() else constant_U();
    }

    // should fail
    procedure condition_is_not_predicate() = {
        if constant_T() then constant_T() else constant_T();
    }
}

concept AssertionTests = {

    require type T;

    predicate pred();

    function func(): T;

    // should succeed
    axiom axiom_with_pred() = { assert pred(); }
    // should fail
    axiom axiom_with_func() = { assert func(); }
}
