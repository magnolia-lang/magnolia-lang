package examples.bgl_v2.mg-src.graph
    imports examples.bgl_v2.mg-src.list;

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
    /*require Edge;

    // TODO: define at least a container relation between edge and edge list
    require type Edge;
    require type Vertex;
    
    use IterableList[ A => EdgeDescriptor
                    , List => EdgeList
                    , ListIterator => EdgeIterator
                    , empty => emptyEdgeList
                    , iterBegin => edgeIterBegin
                    , iterEnd => edgeIterEnd
                    , iterNext => edgeIterNext
                    , iterUnpack => edgeIterUnpack
                    ];
    */
    require type Vertex;

    type EdgeDescriptor;
    type EdgeIterator;
    procedure edgeIterNext(upd ei: EdgeIterator);
    function edgeIterUnpack(ei: EdgeIterator): EdgeDescriptor;

    type Graph;
    type VertexCount;
    type VertexDescriptor;

    procedure outEdges(obs v: VertexDescriptor,
                       obs g: Graph,
                       out itBeg: EdgeIterator,
                       out itEnd: EdgeIterator);
    function outDegree(v: VertexDescriptor, g: Graph): VertexCount;

    // TODO: add axioms such that outDegree(v, g) <= count(outEdges(v, g))?
}

concept VertexListGraph = {
    // Why do we choose a list? And more importantly, what are the
    // requirements that interest us and that we make on a list?
    // Likely:
    //   - the fact that it's a collection of elements
    //   - ?
    /*
    use IterableList[ A => VertexDescriptor
                    , List => VertexList
                    , ListIterator => VertexIterator
                    , empty => emptyVertexList
                    , iterBegin => vertexIterBegin
                    , iterEnd => vertexIterEnd
                    , iterNext => vertexIterNext
                    , iterUnpack => vertexIterUnpack
                    ];
    */
    require type Vertex;
    type Graph;
    type VertexCount;
    type VertexDescriptor;
    type VertexIterator;
    procedure vertexIterNext(upd ei: VertexIterator);
    function vertexIterUnpack(ei: VertexIterator): VertexDescriptor;

    procedure vertices(obs g: Graph,
                       out itBeg: VertexIterator,
                       out itEnd: VertexIterator);
    function numVertices(g: Graph): VertexCount;
}
