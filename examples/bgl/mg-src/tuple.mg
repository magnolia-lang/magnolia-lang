package examples.bgl.mg-src.tuple;

concept Pair = {
    require type A;
    require type B;
    type Pair;

    function first(p: Pair): A;
    function second(p: Pair): B;

    function makePair(a: A, b: B): Pair;

    axiom projectionBehaviorPair(a: A, b: B) {
        var pair = makePair(a, b);
        assert first(pair) == a;
        assert second(pair) == b;
    }
}

concept Triplet = {
    require type A;
    require type B;
    require type C;
    type Triplet;

    function first(p: Triplet): A;
    function second(p: Triplet): B;
    function third(p: Triplet): C;

    function makeTriplet(a: A, b: B, c: C): Triplet;
    
    axiom projectionBehaviorTriplet(a: A, b: B, c: C) {
        var triplet = makeTriplet(a, b, c);
        assert first(triplet) == a;
        assert second(triplet) == b;
        assert third(triplet) == c;
    }
}
