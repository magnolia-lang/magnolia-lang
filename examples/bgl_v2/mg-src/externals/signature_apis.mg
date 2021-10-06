package examples.bgl_v2.mg-src.externals.signature_apis
    imports examples.bgl_v2.mg-src.color_marker
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.list
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.while_loop;

implementation ExtBaseTypes = {
    type Int;
    type Vertex;
}

implementation ExtBaseIntOps = {
    type Int;
    function plus(i1: Int, i2: Int): Int;
    predicate less(i1: Int, i2: Int);
}

implementation ExtColorMarker = signature(ColorMarker);

// TODO: there is a lot of requirements to pass through because we do not have
// the extend keyword. Perhaps this should be improved.
implementation ExtEdge = signature(Edge);

implementation ExtIncidenceAndVertexListGraph = {
    require signature(Edge);
    use IncidenceGraph[ cons => consEdgeList
                      , head => headEdgeList
                      , isEmpty => isEmptyEdgeList
                      , tail => tailEdgeList
                      ];
    use VertexListGraph[ cons => consVertexList
                       , head => headVertexList
                       , isEmpty => isEmptyVertexList
                       , tail => tailVertexList
                       ];

    type Graph;
}

implementation ExtList = signature(List);

implementation ExtReadWritePropertyMapWithInitList = {
    use signature(ReadWritePropertyMap);

    require List[ A => Key
                , List => KeyList
                , empty => emptyKeyList
                ];

    function emptyMap(): PropertyMap;
    function initMap(kl: KeyList, v: Value): PropertyMap;
}

implementation ExtQueue = signature(Queue);
implementation ExtUpdateablePriorityQueue = signature(UpdateablePriorityQueue);

implementation ExtPair = signature(Pair);
implementation ExtTriplet = signature(Triplet);

implementation ExtWhileLoop = signature(WhileLoop);
