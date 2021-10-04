package examples.bgl_v2.mg-src.graph
    imports examples.bgl_v2.mg-src.list;

concept Edge = {
    require type Vertex;
    type Edge;

    function src(e: Edge): Vertex;
    function tgt(e: Edge): Vertex;
}

concept IncidenceGraph = {
    require Edge;

    // TODO: define at least a container relation between edge and edge list
    require type Edge;
    require type Vertex;
    
    require List[ A => Edge
                , List => EdgeList
                , empty => emptyEdgeList
                ];
    type Graph;

    type VertexCount;

    function outEdges(v: Vertex, g: Graph): EdgeList;
    function outDegree(v: Vertex, g: Graph): VertexCount;

    // TODO: add axioms such that outDegree(v, g) <= count(outEdges(v, g))?
}

concept VertexListGraph = {
    // Why do we choose a list? And more importantly, what are the
    // requirements that interest us and that we make on a list?
    // Likely:
    //   - the fact that it's a collection of elements
    //   - ?
    require List[ A => Vertex
                , List => VertexList
                , empty => emptyVertexList
                ];
    type Graph;
    type VertexCount;

    function vertices(g: Graph): VertexList;
    function numVertices(g: Graph): VertexCount;
}
