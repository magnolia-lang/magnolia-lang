package tests.inputs.stmtTests;

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
};
