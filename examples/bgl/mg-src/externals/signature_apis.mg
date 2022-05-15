package examples.bgl.mg-src.externals.signature_apis
    imports examples.bgl.mg-src.bool
          , examples.bgl.mg-src.color_marker
          , examples.bgl.mg-src.for_loop
          , examples.bgl.mg-src.graph
          , examples.bgl.mg-src.list
          , examples.bgl.mg-src.property_map
          , examples.bgl.mg-src.queue
          , examples.bgl.mg-src.stack
          , examples.bgl.mg-src.tuple
          , examples.bgl.mg-src.unit
          , examples.bgl.mg-src.vector
          , examples.bgl.mg-src.while_loop;

implementation ExtBaseTypes = {
    type Int;
    type Vertex;
}

implementation ExtBaseFloatOps = {
    type Float;
    function plus(i1: Float, i2: Float): Float;
    predicate less(i1: Float, i2: Float);
    function negate(f: Float): Float;
    function zero(): Float;
}

implementation ExtColorMarker = signature(ColorMarker);

// TODO: there is a lot of requirements to pass through because we do not have
// the extend keyword. Perhaps this should be improved.
implementation ExtEdge = signature(Edge);

implementation ExtIncidenceAndVertexListAndEdgeListGraph = {
    require type Vertex;

    use Edge;
    use EdgeListGraph;
    use IncidenceGraph;
    use VertexListGraph;

    function toVertexDescriptor(v: Vertex, g: Graph): VertexDescriptor;
    function toEdgeDescriptor(v1: VertexDescriptor,
                              v2: VertexDescriptor,
                              g: Graph): EdgeDescriptor;
}

implementation ExtCustomIncidenceAndVertexListAndEdgeListGraph = {
    require EdgeWithoutDescriptor[ src => srcPlainEdge
                                 , tgt => tgtPlainEdge
                                 ];
    require IterableList[ A => Edge
                        , ListIterator => OutEdgeIterator
                        , List => EdgeList
                        , cons => consEdgeList
                        , empty => emptyEdgeList
                        , getIterator => getOutEdgeIterator
                        , head => headEdgeList
                        , isEmpty => isEmptyEdgeList
                        , iterEnd => outEdgeIterEnd
                        , iterNext => outEdgeIterNext
                        , iterUnpack => outEdgeIterUnpack
                        , tail => tailEdgeList
                        ];

    require IterableList[ A => Edge
                        , ListIterator => EdgeIterator
                        , List => EdgeList
                        , cons => consEdgeList
                        , empty => emptyEdgeList
                        , getIterator => getEdgeIterator
                        , head => headEdgeList
                        , isEmpty => isEmptyEdgeList
                        , iterEnd => edgeIterEnd
                        , iterNext => edgeIterNext
                        , iterUnpack => edgeIterUnpack
                        , tail => tailEdgeList
                        ];

    require type Vertex;
    require IterableList[ A => Vertex
                        , ListIterator => VertexIterator
                        , List => VertexList
                        , cons => consVertexList
                        , empty => emptyVertexList
                        , getIterator => getVertexIterator
                        , head => headVertexList
                        , isEmpty => isEmptyVertexList
                        , iterEnd => vertexIterEnd
                        , iterNext => vertexIterNext
                        , iterUnpack => vertexIterUnpack
                        , tail => tailVertexList
                        ];

    type Graph;
    type VertexCount;

    function src(e: Edge, g: Graph): Vertex;
    function tgt(e: Edge, g: Graph): Vertex;

    procedure outEdges(obs v: Vertex, obs g: Graph,
                       out itr: OutEdgeIterator);
    function outDegree(v: Vertex, g: Graph): VertexCount;

    procedure edges(obs g: Graph, out itr: EdgeIterator);
    procedure vertices(obs g: Graph, out itr: VertexIterator);
    function numVertices(g: Graph): VertexCount;
}

implementation ExtEdgeWithoutDescriptor = signature(EdgeWithoutDescriptor);
/*
implementation ExtCustomIncidenceAndVertexListGraph = {
    require type Vertex;

    use IncidenceGraph;
    use VertexListGraph;

    function toVertexDescripor(v: Vertex, g: Graph): VertexDescriptor;
    function toEdgeDescriptor(v1: VertexDescriptor,
}*/

implementation ExtList = signature(List);

implementation ExtIterableList = signature(IterableList);

implementation ExtReadWriteColorMapWithInitList = {
    use signature(ReadWritePropertyMap)[ PropertyMap => ColorPropertyMap
                                       , Value => Color
                                       ];
    type Color;
    function white(): Color;
    function black(): Color;
    function gray(): Color;

    require type KeyListIterator;
    require predicate iterEnd(kli: KeyListIterator);
    require procedure iterNext(upd kli: KeyListIterator);
    require function iterUnpack(kli: KeyListIterator): Key;

    function initMap(kli: KeyListIterator,
                     v: Color): ColorPropertyMap;
}

implementation ExtReadWritePropertyMapWithInitList = {
    use signature(ReadWritePropertyMap);

    require type KeyListIterator;
    require predicate iterEnd(kli: KeyListIterator);
    require procedure iterNext(upd kli: KeyListIterator);
    require function iterUnpack(kli: KeyListIterator): Key;

    function emptyMap(): PropertyMap;
    function initMap(kli: KeyListIterator,
                     v: Value): PropertyMap;
}

implementation ExtFIFOQueue = signature(FIFOQueue);
implementation ExtUpdateablePriorityQueue = signature(UpdateablePriorityQueue);

implementation ExtStack = signature(Stack);

implementation ExtPair = signature(Pair);
implementation ExtTriplet = signature(Triplet);

implementation ExtVector = signature(Vector);

implementation ExtForIteratorLoop = signature(ForIteratorLoop);
implementation ExtForIteratorLoop1_2 = signature(ForIteratorLoop1_2);
implementation ExtForIteratorLoop1_3 = signature(ForIteratorLoop1_3);
implementation ExtForIteratorLoop2_3 = signature(ForIteratorLoop2_3);
implementation ExtForIteratorLoop3_2 = signature(ForIteratorLoop3_2);

implementation ExtWhileLoop = signature(WhileLoop);
implementation ExtWhileLoop3 = signature(WhileLoop3);
implementation ExtWhileLoop4_3 = signature(WhileLoop4_3);

implementation ExtUnit = signature(Unit);
implementation ExtBool = signature(Bool);
