package examples.bgl.mg-src.v2.graph;

concept Edge = {
    type Edge;
    type Vertex;

    function src(e: Edge): Vertex;
    function tgt(e: Edge): Vertex;
}

concept IncidenceGraph = {
    require Edge;

    // TODO: define at least a container relation between edge and edge list
    type Edge;
    type EdgeList;
    type Vertex;
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
    type VertexList;
    type Graph;
    type VertexCount;

    function vertices(g: Graph): VertexList;
    function numVertices(g: Graph): VertexCount;
}
