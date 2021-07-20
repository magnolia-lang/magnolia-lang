package tests.check.inputs.stateHandlingTests;

implementation I = {
    type T;
    type U;

    // Stateful computation using a value block, should fail
    procedure stateful_ctx_value(obs t: T) { t; value t; }

    // Stateful computation using an effectful block, should succeed
    procedure stateful_ctx_effect(obs t: T) { t; }

    // Value block attempting to return values of different types, should fail
    function value_block_inconsistent_types(t: T, u: U): T {
        value t; value u;
    }

    // Value block attempting to update the outer environment, should fail
    procedure value_block_attempting_side_effect(obs t: T) {
        var out_t: T;
        {
            out_t = t;
            value t;
        };
    }

    function test_fun(in: T): T;
    procedure test_proc(out o: T);

    // Stateful computation when evaluating callable arguments, should fail
    function stateful_callable_argument(): T = {
        var t: T;
        value test_fun(call test_proc(t));
    }

    // Stateful computation when declaring a variable, should fail
    function stateful_variable_declaration(): T = {
        var t: T;
        var y = call test_proc(t);
    }

    // Stateful computation within an assertion, should fail
    function stateful_assertion(): T = {
        var t: T;
        assert call test_proc(t);
    }

    // Access a variable declared in a value block, should fail
    function access_variable_in_value_block(): T = {
        var x = {
            var t: T;
            call test_proc(t);
            value t;
        };
        x = t;
    }
};
