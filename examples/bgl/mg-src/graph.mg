package examples.bgl.mg-src.graph
    imports examples.bgl.mg-src.list;

concept EdgeWithoutDescriptor = {
    require type Vertex;

    type Edge;
    
    function src(e: Edge): Vertex;
    function tgt(e: Edge): Vertex;

    function makeEdge(s: Vertex, t: Vertex): Edge;
}

concept Edge = {
    require type Vertex;
    require type VertexDescriptor;
    type Edge;
    type EdgeDescriptor;
    type Graph;

    function src(e: EdgeDescriptor, g: Graph): VertexDescriptor;
    function tgt(e: EdgeDescriptor, g: Graph): VertexDescriptor;

    function makeEdge(s: Vertex, t: Vertex): Edge;
}

concept IncidenceGraph = {
    require type Vertex;

    type EdgeDescriptor;
    type EdgeIterator;
    procedure edgeIterNext(upd ei: EdgeIterator);
    function edgeIterUnpack(ei: EdgeIterator): EdgeDescriptor;
    predicate edgeIterEnd(ei: EdgeIterator);

    type Graph;
    type VertexCount;
    type VertexDescriptor;

    procedure outEdges(obs v: VertexDescriptor,
                       obs g: Graph,
                       out itr: EdgeIterator);
    function outDegree(v: VertexDescriptor, g: Graph): VertexCount;

    // TODO: add axioms such that outDegree(v, g) <= count(outEdges(v, g))?
}

concept VertexListGraph = {
    require type Vertex;
    type Graph;
    type VertexCount;
    type VertexDescriptor;
    type VertexIterator;
    
    procedure vertexIterNext(upd ei: VertexIterator);
    function vertexIterUnpack(ei: VertexIterator): VertexDescriptor;
    predicate vertexIterEnd(ei: VertexIterator);

    procedure vertices(obs g: Graph,
                       out itr: VertexIterator);
    function numVertices(g: Graph): VertexCount;
}
