package examples.bgl_v2.mg-src.bgl_v2-py
    imports examples.bgl_v2.mg-src.bfs_utils
          , examples.bgl_v2.mg-src.dijkstra_utils
          , examples.bgl_v2.mg-src.externals.python_apis;

program PyBFSTestVisitor = {
    use GenericBFSTestVisitor;

    use PyColorMarker;
    use PyList[ A => Edge
               , List => EdgeList
               , empty => emptyEdgeList
               ];
    use PyList[ A => Vertex
               , List => VertexList
               , empty => emptyVertexList
               ];

    use PyTriplet[ A => VertexList
                  , B => FIFOQueue
                  , C => ColorPropertyMap
                  , Triplet => OuterLoopState
                  , makeTriplet => makeOuterLoopState
                  ];

    use PyPair[ A => OuterLoopState
               , B => EdgeList
               , Pair => InnerLoopState
               , makePair => makeInnerLoopState
               ];

    use PyPair[ A => Graph
               , B => Vertex
               , Pair => InnerLoopContext
               , makePair => makeInnerLoopContext
               ];

    use PyFIFOQueue[ A => Vertex
                    , isEmpty => isEmptyQueue
                    ];
    
    use PyWhileLoop[ Context => Graph
                    , State => OuterLoopState
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];

    use PyWhileLoop[ Context => InnerLoopContext
                    , State => InnerLoopState
                    , cond => bfsInnerLoopCond
                    , step => bfsInnerLoopStep
                    , repeat => bfsInnerLoopRepeat
                    ];

    use PyReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];

    use PyBaseTypes;
    use PyEdge;
    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph[ consEdgeList => cons
                                      , consVertexList => cons
                                      , headEdgeList => head
                                      , headVertexList => head
                                      , isEmptyEdgeList => isEmpty
                                      , isEmptyVertexList => isEmpty
                                      , tailEdgeList => tail
                                      , tailVertexList => tail
                                      ];
};

program PyDijkstraVisitor = {
    use GenericDijkstraVisitor;

    use PyColorMarker;
    use PyList[ A => Edge
               , List => EdgeList
               , empty => emptyEdgeList
               ];
    use PyList[ A => Vertex
               , List => VertexList
               , empty => emptyVertexList
               ];
    use PyList[ A => VertexPair
               , List => VertexPairList
               , empty => emptyVertexPairList
               ];

    use PyTriplet[ A => StateWithMaps
                  , B => PriorityQueue
                  , C => ColorPropertyMap
                  , Triplet => OuterLoopState
                  , makeTriplet => makeOuterLoopState
                  ];

    use PyTriplet[ A => VertexCostMap
                  , B => VertexPredecessorMap
                  , C => EdgeCostMap
                  , Triplet => StateWithMaps
                  , makeTriplet => makeStateWithMaps
                  , first => getVertexCostMap
                  , second => getVertexPredecessorMap
                  , third => getEdgeCostMap
                  ];

    use PyPair[ A => OuterLoopState
               , B => EdgeList
               , Pair => InnerLoopState
               , makePair => makeInnerLoopState
               ];

    use PyPair[ A => Graph
               , B => Vertex
               , Pair => InnerLoopContext
               , makePair => makeInnerLoopContext
               ];

    use PyPair[ A => VertexPredecessorMap
               , B => VertexList
               , Pair => PopulateVPMapState
               ];

    use PyPair[ A => Vertex
               , B => Vertex
               , Pair => VertexPair
               , makePair => makeVertexPair
               ];

    use PyUpdateablePriorityQueue[ A => Vertex
                                  , Priority => Cost
                                  , PriorityMap => VertexCostMap
                                  , empty => emptyPriorityQueue
                                  , isEmpty => isEmptyQueue
                                  ];
    //use PyUpdateablePriorityQueue[ A => CostAndVertex ];
    use PyWhileLoop[ Context => Graph
                    , State => OuterLoopState
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];

    use PyWhileLoop[ Context => InnerLoopContext
                    , State => InnerLoopState
                    , cond => bfsInnerLoopCond
                    , step => bfsInnerLoopStep
                    , repeat => bfsInnerLoopRepeat
                    ];

    use PyWhileLoop[ Context => Vertex
                    , State => PopulateVPMapState
                    , cond => populateVPMapLoopCond
                    , step => populateVPMapLoopStep
                    , repeat => populateVPMapLoopRepeat
                    ];

    use PyReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];

    use PyReadWritePropertyMapWithInitList[ Key => Edge
                                           , KeyList => EdgeList
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyKeyList => emptyEdgeList
                                           , emptyMap => emptyECMap
                                           ];

    use PyReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Vertex
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVPMap
                                           ];

    use PyReadWritePropertyMapWithInitList[ Key => Vertex
                                          , KeyList => VertexList
                                          , Value => Cost
                                          , PropertyMap => VertexCostMap
                                          , emptyKeyList => emptyVertexList
                                          , emptyMap => emptyVCMap
                                          ];

    use PyBaseTypes;
    use PyBaseFloatOps[ Float => Cost ];
    use PyEdge;
    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph[ consEdgeList => cons
                                     , consVertexList => cons
                                     , headEdgeList => head
                                     , headVertexList => head
                                     , isEmptyEdgeList => isEmpty
                                     , isEmptyVertexList => isEmpty
                                     , tailEdgeList => tail
                                     , tailVertexList => tail
                                     ];
}
