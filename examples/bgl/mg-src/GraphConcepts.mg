package examples.bgl.mg-src.GraphConcepts
    imports examples.bgl.mg-src.BasicConcepts;

concept Edge = {
    type Vertex;
    type Edge;

    function source(e: Edge): Vertex;
    function target(e: Edge): Vertex;
}

concept IncidenceGraph = {
    type Graph;
    
    use Edge[ Edge => Edge
            , Vertex => Vertex
            , source => source
            , target => target
            ];

    use Collection[ Collection => EdgeCollection
                  , E => Edge
                  , isIn => isIn
                  ];

    function outEdges(g: Graph, v: Vertex): EdgeCollection;
}

// TODO: implement different kinds of collections, and swap them in programs.
concept AdjacencyGraph = {
    type Graph;
    type VertexCollection;
    type Vertex;

    function adjacentVertices(g: Graph, v: Vertex): VertexCollection;
}

