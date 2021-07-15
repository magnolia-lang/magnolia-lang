package tests.self-contained-codegen.inputs.generalTranslationTests;

implementation MgImpl = external C++ E.ExternalImplementation {
    type X;
    type Y;
    function mkX(): X;
    function mkY(): Y;
    // TODO(bchetioui, 20/07/21): disallow such names in externals, as this is
    // not a valid C++ identifier. We let this pass to not overload the PR.
    function _+_(lhs: X, rhs: X): X;
};

implementation Helper = {
    type X;

    require function mkX(): X;
    procedure helper_assign(out x_out: X) {
        x_out = mkX();
    }
};

program P = {
    use MgImpl[mkY => _mkY];
    use Helper;

    procedure all_statements(obs x_obs: X, upd x_upd: X, out x_out: X) {
        // simple assignment
        x_out = x_obs;

        // assignment of a value block
        x_out = { value mkX(); };

        // variable declaration without assignment
        var x_var_unassigned: X;

        // variable declaration with assignment
        var x_var_assigned = x_obs;

        // assertion
        assert x_obs == x_obs;

        // call procedure
        call helper_assign(x_upd);

        // if-then-else expression
        x_out = if x_obs == x_obs then x_out else x_obs;

        // if-then-else statement
        if x_obs == x_obs then {} else {};
    }

    // overloading on return types
    function some_f_overloaded_on_return_type(): X = mkX();
    function some_f_overloaded_on_return_type(): Y = _mkY();

    // templated call to overloaded function
    function call_f_overloaded_on_return_type(): X =
        some_f_overloaded_on_return_type();
};
