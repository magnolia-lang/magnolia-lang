package examples.bgl_v2.mg-src.bgl_v2-cpp
    imports examples.bgl_v2.mg-src.bfs_utils
          , examples.bgl_v2.mg-src.dijkstra_utils
          , examples.bgl_v2.mg-src.externals.cpp_apis;

program CppBFSTestVisitor = {
    use GenericBFSTestVisitor;

    use CppColorMarker;
    use CppList[ A => Edge
               , List => EdgeList
               , empty => emptyEdgeList
               ];
    use CppList[ A => Vertex
               , List => VertexList
               , empty => emptyVertexList
               ];

    use CppTriplet[ A => VertexList
                  , B => FIFOQueue
                  , C => ColorPropertyMap
                  , Triplet => OuterLoopState
                  , makeTriplet => makeOuterLoopState
                  ];

    use CppPair[ A => OuterLoopState
               , B => EdgeList
               , Pair => InnerLoopState
               , makePair => makeInnerLoopState
               ];

    use CppPair[ A => Graph
               , B => Vertex
               , Pair => InnerLoopContext
               , makePair => makeInnerLoopContext
               ];

    use CppFIFOQueue[ A => Vertex
                    , isEmpty => isEmptyQueue
                    ];
    
    use CppWhileLoop[ Context => Graph
                    , State => OuterLoopState
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];

    use CppWhileLoop4[ Context => InnerLoopContext
                     , State1 => VertexList
                     , State2 => FIFOQueue
                     , State3 => ColorPropertyMap
                     , State4 => EdgeList
                     , cond => bfsInnerLoopCond
                     , step => bfsInnerLoopStep
                     , repeat => bfsInnerLoopRepeat
                     ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];

    use CppBaseTypes;
    use CppEdge;
    // CppIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListGraph[ consEdgeList => cons
                                      , consVertexList => cons
                                      , headEdgeList => head
                                      , headVertexList => head
                                      , isEmptyEdgeList => isEmpty
                                      , isEmptyVertexList => isEmpty
                                      , tailEdgeList => tail
                                      , tailVertexList => tail
                                      ];
};

program CppDijkstraVisitor = {
    use GenericDijkstraVisitor;

    use CppColorMarker;
    use CppList[ A => Edge
               , List => EdgeList
               , empty => emptyEdgeList
               ];
    use CppList[ A => Vertex
               , List => VertexList
               , empty => emptyVertexList
               ];
    use CppList[ A => VertexPair
               , List => VertexPairList
               , empty => emptyVertexPairList
               ];

    use CppTriplet[ A => StateWithMaps
                  , B => PriorityQueue
                  , C => ColorPropertyMap
                  , Triplet => OuterLoopState
                  , makeTriplet => makeOuterLoopState
                  ];

    use CppTriplet[ A => VertexCostMap
                  , B => VertexPredecessorMap
                  , C => EdgeCostMap
                  , Triplet => StateWithMaps
                  , makeTriplet => makeStateWithMaps
                  , first => getVertexCostMap
                  , second => getVertexPredecessorMap
                  , third => getEdgeCostMap
                  ];

    use CppPair[ A => OuterLoopState
               , B => EdgeList
               , Pair => InnerLoopState
               , makePair => makeInnerLoopState
               ];

    use CppPair[ A => Graph
               , B => Vertex
               , Pair => InnerLoopContext
               , makePair => makeInnerLoopContext
               ];

    use CppPair[ A => VertexPredecessorMap
               , B => VertexList
               , Pair => PopulateVPMapState
               ];

    use CppPair[ A => Vertex
               , B => Vertex
               , Pair => VertexPair
               , makePair => makeVertexPair
               ];

    use CppUpdateablePriorityQueue[ A => Vertex
                                  , Priority => Cost
                                  , PriorityMap => VertexCostMap
                                  , empty => emptyPriorityQueue
                                  , isEmpty => isEmptyQueue
                                  ];
    //use CppUpdateablePriorityQueue[ A => CostAndVertex ];
    use CppWhileLoop[ Context => Graph
                    , State => OuterLoopState
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];
    
    use CppWhileLoop4[ Context => InnerLoopContext
                     , State1 => StateWithMaps
                     , State2 => PriorityQueue
                     , State3 => ColorPropertyMap
                     , State4 => EdgeList
                     , cond => bfsInnerLoopCond
                     , step => bfsInnerLoopStep
                     , repeat => bfsInnerLoopRepeat
                     ];

    use CppWhileLoop[ Context => Vertex
                    , State => PopulateVPMapState
                    , cond => populateVPMapLoopCond
                    , step => populateVPMapLoopStep
                    , repeat => populateVPMapLoopRepeat
                    ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Edge
                                           , KeyList => EdgeList
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyKeyList => emptyEdgeList
                                           , emptyMap => emptyECMap
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Vertex
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVPMap
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVCMap
                                           ];

    use CppBaseTypes;
    use CppBaseFloatOps[ Float => Cost ];
    use CppEdge;
    // CppIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListGraph[ consEdgeList => cons
                                      , consVertexList => cons
                                      , headEdgeList => head
                                      , headVertexList => head
                                      , isEmptyEdgeList => isEmpty
                                      , isEmptyVertexList => isEmpty
                                      , tailEdgeList => tail
                                      , tailVertexList => tail
                                      ];
}
