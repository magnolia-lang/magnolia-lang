package examples.bgl.mg-src.ExternalDataStructures
    imports examples.bgl.mg-src.BasicConcepts;

implementation CxxQueue = external C++ base.queue {
    type Queue;
    require type E;

    function nil(): Queue;
    function enqueue(q: Queue, e: E): Queue;
    function dequeue(q: Queue): Queue guard !empty(q);
    function first(q: Queue): E guard !empty(q);

    predicate empty(q: Queue);
}

implementation CxxList = external C++ base.list {
    type List;
    require type E;

    function nil(): List;
    function append(l: List, e: E): List;
    function head(l: List): E guard !empty(l);
    function tail(l: List): List guard !empty(l);

    predicate empty(l: List);
}

// TODO: add keyword extend, which I think would really help here.
implementation CxxMapFunction = external C++ base.map_function {
    require CollectionMapFunction;

    function map(dc: CollectionA): CollectionB;
    require function f(c: A): B;

    // extra functions since we can not extend
    require function emptyCollectionB(): CollectionB;
    require function addToCollection(cb: CollectionB, b: B): CollectionB;
    require function extractOneElement(ca: CollectionA): A;
    require predicate isCollectionEmpty(ca: CollectionA);
    require function removeFromCollection(ca: CollectionA, a: A): CollectionA;
}

// TODO: same as require GraphConcepts.Edge + concrete defs
implementation CxxEdge = external C++ base.edge {
    require type Vertex;
    type Edge;

    function source(e: Edge): Vertex;
    function target(e: Edge): Vertex;
    function make_edge(src: Vertex, tgt: Vertex): Edge;
}

// TODO: add keyword extend, which I think would really help here.
implementation CxxIncidenceGraph = external C++ base.incidence_graph {
    // this should be 'extend'ed from CxxEdge
    require CxxEdge;
    //type Edge;
    type Graph;
   
    // this should be 'extend'ed from CxxHashSet, so that ops are provided
    // it should also be possible to iterate over it. We should specify
    // that through an extent, perhaps.
    require CxxHashSet[ HashSet => EdgeCollection
                      , E => Edge
                      , insert => addToCollection
                      , remove => removeFromCollection
                      , nil => emptyEdgeCollection
                      , min => extractOneElement
                      , member => isIn
                      , empty => isCollectionEmpty
                      ];

    function outEdges(g: Graph, v: Vertex): EdgeCollection;
}

implementation CxxAdjacencyGraph = external C++ base.adjacency_graph {
    type Graph;
    require type VertexCollection;
    require type Vertex;

    function adjacentVertices(g: Graph, v: Vertex): VertexCollection;
}

implementation CxxHashSet = external C++ base.hash_set {
    type HashSet;
    require type E;

    function insert(h: HashSet, e: E): HashSet;
    function remove(h: HashSet, e: E): HashSet;
    function nil(): HashSet;
    function min(h: HashSet): E guard !empty(h);

    predicate member(h: HashSet, e: E);
    predicate empty(h: HashSet);
}

// simply a type to use as element
implementation CxxString = external C++ base.string {
    type String;
}

implementation CxxPair = external C++ base.pair {
    type Pair;
    require type A;
    require type B;

    function make_pair(a: A, b: B): Pair;
    function first(p: Pair): A;
    function second(p: Pair): B;
}

implementation CxxTuple3 = external C++ base.tuple_3 {
    type Tuple;
    require type A;
    require type B;
    require type C;

    function make_tuple(a: A, b: B, c: C): Tuple;
    function first(t: Tuple): A;
    function second(t: Tuple): B;
    function third(t: Tuple): C;
}

implementation CxxPPrinter = external C++ base.pprinter {
    require type E;

    procedure pprint(obs e: E); // TODO: we do not carry the stream here,
                                //       for debugging reasons (we should)
}

implementation CxxUnit = external C++ base.unit {
    type Unit; // use Unit to replace streams for now
}

implementation CxxWhileLoop = external C++ base.while_loop {
    require type State;
    require type Context;

    require predicate cond(state: State, ctx: Context);
    require procedure body(upd state: State, obs ctx: Context);

    procedure repeat(upd state: State, obs ctx: Context);
}

