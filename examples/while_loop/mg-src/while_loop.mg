package examples.while_loop.mg-src.while_loop;

implementation WhileLoop = external C++ base.while_ops {
    require type Context;
    require type State;
    require predicate cond(s: State, c: Context);
    require procedure body(upd s: State, obs c: Context);
    procedure repeat(upd s: State, obs c: Context);
}

implementation Utils = {
    require type IntLike;
    require function add(a: IntLike, b: IntLike): IntLike;
    require function one(): IntLike;

    procedure increment(upd counter: IntLike, obs bound: IntLike) = {
        counter = add(counter, one());
    }
}

signature IntLikeOps = {
    type IntLike;

    function one(): IntLike;
    function add(a: IntLike, b: IntLike): IntLike;
    predicate isLowerThan(a: IntLike, b: IntLike);
}

// TODO: write with a signature cast & renamings
implementation Int16Utils = external C++ base.int16_utils IntLikeOps[IntLike => Int16];

implementation Int32Utils = external C++ base.int32_utils IntLikeOps[IntLike => Int32];

program IterationProgram = {
    use Int16Utils;
    use Int32Utils;

    use Utils[IntLike => Int16];
    use Utils[IntLike => Int32];

    use WhileLoop[ Context => Int32
                 , State => Int32
                 , cond => isLowerThan
                 , body => increment
                 ];
    use WhileLoop[ Context => Int16
                 , State => Int16
                 , cond => isLowerThan
                 , body => increment
                 ];
}
