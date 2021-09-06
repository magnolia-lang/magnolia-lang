package tests.serialization.inputs.onePackageTests;

renaming Renaming = [U => T]

implementation ExternalModule = external C++ path.to.file.struct {
    // an external concrete type
    type U;
    // an external callable implementation
    function external_f(): U;
}

implementation LocalModule = {
    // a local abstract type
    type T;

    // a dependency with both one total and one partial renaming
    use ExternalModule[Renaming][[T => U, U => U]];

    // a local function prototype
    function localAbstractFunction(): T;
    // a local procedure prototype
    procedure localAbstractProcedure(upd t: T);

    // a local callable implementation
    procedure localConcreteCallable(obs t_obs: T, out t_out: T, upd t_upd: T) {
        // simple assignment
        t_out = t_obs;

        // assignment of a value block
        t_out = { value localAbstractFunction(); };

        // variable declaration without assignment
        var t_var_unassigned: T;

        // variable declaration with assignment
        var t_var_assigned = t_obs;

        // assertion
        assert t_obs == t_obs;

        // call procedure
        call localAbstractProcedure(t_upd);

        // if-then-else expression
        t_out = if t_obs == t_obs then t_out else t_obs;

        // if-then-else statement
        if t_obs == t_obs then {} else {};

        // true, false constants
        if TRUE() || FALSE() then {} else {};
    }
}

