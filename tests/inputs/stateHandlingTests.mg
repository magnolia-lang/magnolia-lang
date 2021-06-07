package tests.inputs.stateHandlingTests;

implementation I = {
    type T;
    type U;

    // Stateful computation using a value block, should fail
    procedure stateful_ctx_value(obs t: T) { t; value t; }

    // Stateful computation using an effectful block, should succeed
    procedure stateful_ctx_effect(obs t: T) { t; }

    // Value block attempting to update the outer environment, should fail
    procedure value_block_attempting_side_effect(obs t: T) {
        var out_t: T;
        {
            out_t = t;
            value t;
        };
    }

    // Value block attempting to return values of different types, should fail
    function value_block_inconsistent_types(t: T, u: U): T {
        value t; value u;
    }
};
