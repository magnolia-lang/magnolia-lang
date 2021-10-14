package examples.bgl_v2.mg-src.externals.signature_apis
    imports examples.bgl_v2.mg-src.color_marker
          , examples.bgl_v2.mg-src.for_loop
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.list
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.vector
          , examples.bgl_v2.mg-src.while_loop;

implementation ExtBaseTypes = {
    type Int;
    type Vertex;
}

implementation ExtBaseFloatOps = {
    type Float;
    function plus(i1: Float, i2: Float): Float;
    predicate less(i1: Float, i2: Float);
}

implementation ExtColorMarker = signature(ColorMarker);

// TODO: there is a lot of requirements to pass through because we do not have
// the extend keyword. Perhaps this should be improved.
//implementation ExtEdge = signature(Edge);

implementation ExtIncidenceAndVertexListGraph = {
    require type Vertex;

    use Edge;
    use IncidenceGraph;
    use VertexListGraph;

    function toVertexDescriptor(v: Vertex, g: Graph): VertexDescriptor;
    function toEdgeDescriptor(v1: VertexDescriptor,
                              v2: VertexDescriptor,
                              g: Graph): EdgeDescriptor;
}

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
    require procedure iterNext(upd kli: KeyListIterator);
    require function iterUnpack(kli: KeyListIterator): Key;

    function initMap(klBeg: KeyListIterator,
                     klEnd: KeyListIterator,
                     v: Color): ColorPropertyMap;
}

implementation ExtReadWritePropertyMapWithInitList = {
    use signature(ReadWritePropertyMap);

    require type KeyListIterator;
    require procedure iterNext(upd kli: KeyListIterator);
    require function iterUnpack(kli: KeyListIterator): Key;

    function emptyMap(): PropertyMap;
    function initMap(klBeg: KeyListIterator,
                     klEnd: KeyListIterator,
                     v: Value): PropertyMap;
}

implementation ExtFIFOQueue = signature(FIFOQueue);
implementation ExtUpdateablePriorityQueue = signature(UpdateablePriorityQueue);

implementation ExtPair = signature(Pair);
implementation ExtTriplet = signature(Triplet);

implementation ExtVector = signature(Vector);

implementation ExtForIteratorLoop = signature(ForIteratorLoop);
implementation ExtForIteratorLoop3_2 = signature(ForIteratorLoop3_2);

implementation ExtWhileLoop = signature(WhileLoop);
implementation ExtWhileLoop3 = signature(WhileLoop3);
implementation ExtWhileLoop4_3 = signature(WhileLoop4_3);
