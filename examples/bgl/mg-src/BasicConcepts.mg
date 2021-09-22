package examples.bgl.mg-src.BasicConcepts;

concept Queue = {
    type Queue;
    type E;

    function enqueue(q: Queue, e: E): Queue;
    function dequeue(q: Queue): Queue guard !isQueueEmpty(q);
    function first(q: Queue): E;

    function emptyQueue(): Queue;

    predicate isQueueEmpty(q: Queue);
}

concept WhileLoop = {
    type State;
    type Context;

    predicate condition(st: State, ctx: Context);
    procedure repeat(upd st: State, obs ctx: Context);
    procedure step(upd st: State, obs ctx: Context);

    // TODO: fix bug of stripping axiom bodies
    /*axiom repeatAxiom(st: State, ctx: Context) {
        assert condition(st, ctx) => repeat(st, ctx) == repeat(step(st, ctx), ctx);
        assert !condition(st, ctx) => repeat(st, ctx) == st;
    }*/
}

concept Twople = {
    type A;
    type B;
    type Twople;

    function twople(a: A, b: B): Twople;
    function fst(t: Twople): A;
    function snd(t: Twople): B;
}

concept Triple = {
    type A;
    type B;
    type C;
    type Triple;

    function triple(a: A, b: B, c: C): Triple;
    function fst(t: Triple): A;
    function snd(t: Triple): B;
    function third(t: Triple): C;
}

concept Collection = {
    type Collection;
    type E;

    predicate isIn(c: Collection, e: E);
}

/*
    function emptyCollection(): Collection;
    function insert(c: Collection, e: E): Collection;
    function remove(c: Collection, e: E): Collection;
}*/

concept IterableCollection = {
    type Collection;
    type E;

    function emptyCollection(): Collection;
    function addToCollection(c: Collection, e: E): Collection;

    function extractOneElement(c: Collection): E
        guard !isCollectionEmpty(c);
    function removeFromCollection(c: Collection, e: E): Collection
        guard isIn(c, e);

    predicate isCollectionEmpty(c: Collection);
    predicate isIn(c: Collection, e: E);
}


concept Map = {
    type Key;
    type Value;
    type Map;

    function emptyMap(): Map;
    function elementAt(m: Map, k: Key): Value guard isIn(m, k);
    function insert(m: Map, k: Key, v: Value): Map;

    predicate isIn(m: Map, k: Key);

    axiom insertionAddsElementsToMap(m: Map, k: Key, v: Value) {
        var mapPostInsertion = insert(m, k, v);
        assert isIn(mapPostInsertion, k);
        assert elementAt(mapPostInsertion, k) == v;
    }

    axiom emptyIsEmpty(k: Key) {
        assert !isIn(emptyMap(), k);
    }
}

concept CollectionMapFunction = {
    type A;
    type B;

    type CollectionA;
    type CollectionB;

    function map(dc: CollectionA): CollectionB;
    function f(c: A): B;

    /* Axioms for map require additional functions and relations between
       A/CollectionA, B/CollectionB, but also
       CollectionA and CollectionB. The latter two must have some kind
       of isomorphism between them (I think). Because all types in Magnolia
       are opaque and relations between types can only be explained through
       functions, dealing with several container types is rather painful.
     */
}
