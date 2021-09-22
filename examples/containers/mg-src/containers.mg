package examples.containers.mg-src.containers;

implementation Pair = external C++ base.pair {
    require type A;
    require type B;
    type Pair;

    function make_pair(a: A, b: B): Pair;
    function first(t: Pair): A;
    function second(t: Pair): B;
}

implementation Types = external C++ base.types {
    type Int16;
    type Int32;
}

program NestedPairProgram = {
    use Types;
    use Pair[ Pair => InnerPair
            , A => Int16
            , B => Int32
            ];
    use Pair[ Pair => OuterPair
            , A => InnerPair
            , B => InnerPair
            ];
}
