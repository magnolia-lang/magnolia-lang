package tests.check.inputs.renamingTests;

implementation ToRename = {
    require type T1;
    require type T2;

    require function _t1(): T1;
    require function _t2(): T2;
    require function some_function(t1_in: T1, t2_in: T2): T2;
    procedure some_procedure(obs t1_in: T1, obs t2_in: T2, out t2_out: T2) {
        t2_out = some_function(t1_in, t2_in);
        var var_t1: T1 = _t1();
        var var_t2: T2 = _t2();
    }
}


implementation TypesToT1 = {
    use ToRename[T1 => T1, T2 => T1, _t1 => _t11, _t2 => _t12];

    predicate eq_t11_t12() = _t11() == _t12(); // should succeed
}


implementation SwapTypes = {
    use ToRename[T1 => T2, T2 => T1, _t1 => __t2, _t2 => __t1];

    function _t2(): T2 = __t2(); // should succeed
    function _t1(): T1 = __t1(); // should succeed
}


implementation SwapTypesMultipleSteps = {
    use ToRename[T1 => T3][T2 => T1][T3 => T2][_t1 => __t2, _t2 => __t1];

    function _t2(): T2 = __t2(); // should succeed
    function _t1(): T1 = __t1(); // should succeed
}

implementation NonExistentSourcesInTotalAndPartialRenaming = {
    use ToRename[Z => Z]; // should fail
    use ToRename[[Z => Z]]; // should succeed
}

implementation ToRenameExt = external C++ ToRenameExtCxx {
    type T;
    function f(t: T): T;
}


implementation RenamingOverload = {
    use ToRenameExt[f => _f, T => _T];
    type _T;
    function _f(_t: _T): _T = _t; // should fail
}

signature RenamedModuleExprDef = {
    type T;
}[T => T1];

signature CheckRenamedModuleExprDef = {
    use RenamedModuleExprDef;
    function f(): T1; // should succeed
    function g(): T;  // should fail
}

signature RenamedModuleExprRef = RenamedModuleExprDef[T1 => T2];

signature CheckRenamedModuleExprRef = {
    use RenamedModuleExprRef;
    function f(): T2; // should succeed
    function g(): T1; // should fail
}
