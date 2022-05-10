package examples.bgl.mg-src.bgl-cpp
    imports examples.bgl.mg-src.bellman_ford_utils
          , examples.bgl.mg-src.bfs_utils
          , examples.bgl.mg-src.dfs_utils
          , examples.bgl.mg-src.dijkstra_utils
          , examples.bgl.mg-src.johnson_utils
          , examples.bgl.mg-src.prim_utils
          , examples.bgl.mg-src.externals.cpp_apis;

program CppBFSTestVisitor = {
    use GenericBFSTestVisitor;

    use CppFIFOQueue[ A => VertexDescriptor
                    , isEmpty => isEmptyQueue
                    ];

    use CppWhileLoop3[ Context => Graph
                     , State1 => VertexVector
                     , State2 => FIFOQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use CppForIteratorLoop3_2[ Iterator => OutEdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => VertexVector
                             , State2 => FIFOQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => outEdgeIterEnd
                             , iterNext => outEdgeIterNext
                             , step => bfsInnerLoopStep
                             , forLoopRepeat => bfsInnerLoopRepeat
                             ];


    use CppReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use CppBaseTypes;

    // CppIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListAndEdgeListGraph;

    use CppPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use CppVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};


program CppDFSTestVisitor = {
    use GenericDFSTestVisitor;

    use CppStack[ A => VertexDescriptor
                , isEmpty => isEmptyStack
                ];

    use CppWhileLoop3[ Context => Graph
                     , State1 => VertexVector
                     , State2 => Stack
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use CppForIteratorLoop3_2[ Iterator => OutEdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => VertexVector
                             , State2 => Stack
                             , State3 => ColorPropertyMap
                             , iterEnd => outEdgeIterEnd
                             , iterNext => outEdgeIterNext
                             , step => bfsInnerLoopStep
                             , forLoopRepeat => bfsInnerLoopRepeat
                             ];


    use CppReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use CppBaseTypes;

    // CppIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListAndEdgeListGraph;

    use CppPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use CppVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};


program CppParallelBFSTestVisitor = {
    use GenericBFSTestVisitor;

    use CppThreadSafeFIFOQueue[ A => VertexDescriptor
                              , isEmpty => isEmptyQueue
                              ];

    use CppWhileLoop3[ Context => Graph
                     , State1 => VertexVector
                     , State2 => FIFOQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    // TODO: technically, we only need to have an iterNext expression. In
    // practice, for the sake of the example, we use openmp in the backend for
    // our example, which requires the ability to write ++itr instead of
    // iterNext(itr).
    use CppForParallelIteratorLoop3_2[ Iterator => OutEdgeIterator
                                     , Context1 => Graph
                                     , Context2 => VertexDescriptor
                                     , State1 => VertexVector
                                     , State2 => FIFOQueue
                                     , State3 => ColorPropertyMap
                                     , iterEnd => outEdgeIterEnd
                                     // must be equivalent to incrementation
                                     , iterNext => outEdgeIterNext
                                     , step => bfsInnerLoopStep
                                     , forLoopRepeat => bfsInnerLoopRepeat
                                     ];


    use CppReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use CppBaseTypes;

    // CppIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListAndEdgeListGraph;

    use CppPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use CppThreadSafeVector[ A => VertexDescriptor
                           , Vector => VertexVector
                           , empty => emptyVertexVector
                           ];
};

program CppDijkstraVisitor = {
    use GenericDijkstraVisitor;

    use CppBaseFloatOps[ Float => Cost ];

    use CppUpdateablePriorityQueue[ A => VertexDescriptor
                                  , Priority => Cost
                                  , PriorityMap => VertexCostMap
                                  , empty => emptyPriorityQueue
                                  , isEmpty => isEmptyQueue
                                  ];

    use CppWhileLoop3[ Context => Graph
                     , State1 => StateWithMaps
                     , State2 => PriorityQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use CppForIteratorLoop3_2[ Iterator => OutEdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => StateWithMaps
                             , State2 => PriorityQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => outEdgeIterEnd
                             , iterNext => outEdgeIterNext
                             , step => bfsInnerLoopStep
                             , forLoopRepeat => bfsInnerLoopRepeat
                             ];

    use CppForIteratorLoop[ Context => VertexDescriptor
                          , Iterator => VertexIterator
                          , State => VertexPredecessorMap
                          , iterEnd => vertexIterEnd
                          , iterNext => vertexIterNext
                          , step => populateVPMapLoopStep
                          , forLoopRepeat => populateVPMapLoopRepeat
                          ];

    use CppReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];


    use CppReadWritePropertyMapWithInitList[ Key => EdgeDescriptor
                                           , KeyListIterator => OutEdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyMap => emptyECMap
                                           , iterEnd => outEdgeIterEnd
                                           , iterNext => outEdgeIterNext
                                           , iterUnpack => outEdgeIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => VertexDescriptor
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyMap => emptyVPMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyMap => emptyVCMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
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

    use CppBaseTypes;

    // CppIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListAndEdgeListGraph;

    use CppPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use CppVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};

program CppCustomGraphTypeBFSTestVisitor = {
    use GenericBFSTestVisitor[ VertexDescriptor => Vertex
                             , EdgeDescriptor => Edge
                             ];

    //use CppColorMarker;
    use CppIterableList[ A => Edge
                       , List => EdgeList
                       , ListIterator => OutEdgeIterator
                       , empty => emptyEdgeList
                       , getIterator => getOutEdgeIterator
                       , iterEnd => outEdgeIterEnd
                       , iterNext => outEdgeIterNext
                       , iterUnpack => outEdgeIterUnpack
                       ];
    use CppIterableList[ A => Edge
                       , List => EdgeList
                       , ListIterator => EdgeIterator
                       , empty => emptyEdgeList
                       , getIterator => getEdgeIterator
                       , iterEnd => edgeIterEnd
                       , iterNext => edgeIterNext
                       , iterUnpack => edgeIterUnpack
                       ];
    use CppIterableList[ A => Vertex
                       , List => VertexList
                       , ListIterator => VertexIterator
                       , empty => emptyVertexList
                       , getIterator => getVertexIterator
                       , iterEnd => vertexIterEnd
                       , iterNext => vertexIterNext
                       , iterUnpack => vertexIterUnpack
                       ];

    use CppFIFOQueue[ A => Vertex
                    , isEmpty => isEmptyQueue
                    ];

    use CppWhileLoop3[ Context => Graph
                     , State1 => VertexVector
                     , State2 => FIFOQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use CppForIteratorLoop3_2[ Iterator => OutEdgeIterator
                             , Context1 => Graph
                             , Context2 => Vertex
                             , State1 => VertexVector
                             , State2 => FIFOQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => outEdgeIterEnd
                             , iterNext => outEdgeIterNext
                             , step => bfsInnerLoopStep
                             , forLoopRepeat => bfsInnerLoopRepeat
                             ];

    use CppReadWriteColorMapWithInitList[ Key => Vertex
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use CppBaseTypes;
    use CppEdgeWithoutDescriptor;
    // CppIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppCustomIncidenceAndVertexListGraph[ consEdgeList => cons
                                            , consVertexList => cons
                                            , headEdgeList => head
                                            , headVertexList => head
                                            , isEmptyEdgeList => isEmpty
                                            , isEmptyVertexList => isEmpty
                                            , tailEdgeList => tail
                                            , tailVertexList => tail
                                            ];
    use CppPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use CppVector[ A => Vertex
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};

program CppPrimVisitor = {
    use GenericPrimVisitor;

    use CppBaseFloatOps[ Float => Cost ];

    use CppUpdateablePriorityQueue[ A => VertexDescriptor
                                  , Priority => Cost
                                  , PriorityMap => VertexCostMap
                                  , empty => emptyPriorityQueue
                                  , isEmpty => isEmptyQueue
                                  ];

    use CppWhileLoop3[ Context => Graph
                     , State1 => StateWithMaps
                     , State2 => PriorityQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use CppForIteratorLoop3_2[ Iterator => OutEdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => StateWithMaps
                             , State2 => PriorityQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => outEdgeIterEnd
                             , iterNext => outEdgeIterNext
                             , step => bfsInnerLoopStep
                             , forLoopRepeat => bfsInnerLoopRepeat
                             ];

    use CppForIteratorLoop[ Context => VertexDescriptor
                          , Iterator => VertexIterator
                          , State => VertexPredecessorMap
                          , iterEnd => vertexIterEnd
                          , iterNext => vertexIterNext
                          , step => populateVPMapLoopStep
                          , forLoopRepeat => populateVPMapLoopRepeat
                          ];

    use CppReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];


    use CppReadWritePropertyMapWithInitList[ Key => EdgeDescriptor
                                           , KeyListIterator => OutEdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyMap => emptyECMap
                                           , iterEnd => outEdgeIterEnd
                                           , iterNext => outEdgeIterNext
                                           , iterUnpack => outEdgeIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => VertexDescriptor
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyMap => emptyVPMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyMap => emptyVCMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
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

    use CppBaseTypes;

    // CppIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListAndEdgeListGraph;

    use CppPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use CppVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};

program CppBellmanFord = {
    use GenericBellmanFord;

    use CppBaseFloatOps[ Float => Cost ];

    use CppForIteratorLoop[ Context => Unit
                          , Iterator => VertexIterator
                          , State => VertexPredecessorMap
                          , iterEnd => vertexIterEnd
                          , iterNext => vertexIterNext
                          , step => populateVPMapLoopStep
                          , forLoopRepeat => populateVPMapLoopRepeat
                          ];

    use CppForIteratorLoop3_2[ Context1 => EdgeCostMap
                             , Context2 => Graph
                             , Iterator => VertexIterator
                             , State1 => Unit
                             , State2 => VertexCostMap
                             , State3 => VertexPredecessorMap
                             , iterEnd => vertexIterEnd
                             , iterNext => vertexIterNext
                             , step => edgeRelaxationOuterLoopStep
                             , forLoopRepeat => edgeRelaxationOuterLoopRepeat
                             ];

    use CppForIteratorLoop3_2[ Context1 => EdgeCostMap
                             , Context2 => Graph
                             , Iterator => EdgeIterator
                             , State1 => Unit
                             , State2 => VertexCostMap
                             , State3 => VertexPredecessorMap
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
                             , step => edgeRelaxationInnerLoopStep
                             , forLoopRepeat => edgeRelaxationInnerLoopRepeat
                             ];

    use CppForIteratorLoop2_3[ Context1 => VertexCostMap
                             , Context2 => EdgeCostMap
                             , Context3 => Graph
                             , Iterator => EdgeIterator
                             , State1 => Unit
                             , State2 => Bool
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
                             , step => checkNegativeCycleLoopStep
                             , forLoopRepeat => checkNegativeCycleLoopRepeat
                             ];

    use CppReadWritePropertyMapWithInitList[ Key => EdgeDescriptor
                                           , KeyListIterator => OutEdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyMap => emptyECMap
                                           , iterEnd => outEdgeIterEnd
                                           , iterNext => outEdgeIterNext
                                           , iterUnpack => outEdgeIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => VertexDescriptor
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyMap => emptyVPMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyMap => emptyVCMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    use CppBaseTypes;

    use CppIncidenceAndVertexListAndEdgeListGraph;

    use CppBool;
    use CppUnit;
}

program CppJohnson = {
    use GenericJohnson[ WriteableEdgeCostMap => EdgeCostMap
                      , emptyWECMap => emptyECMap
                      ];

    use CppDijkstraVisitor;
    use CppBellmanFord;

    // VertexCostMatrix
    use CppReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => VertexCostMap
                                           , PropertyMap => VertexCostMatrix
                                           , emptyMap => emptyVCMatrix
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    // Vertex cost initialization
    use CppForIteratorLoop[ Context => Cost
                          , Iterator => VertexIterator
                          , State => VertexCostMap
                          , iterEnd => vertexIterEnd
                          , iterNext => vertexIterNext
                          , step => initializeVertexCostMapLoopStep
                          , forLoopRepeat => initializeVertexCostMapLoopRepeat
                          ];

    // Reweighting
    use CppForIteratorLoop1_3[ Context1 => EdgeCostMap
                             , Context2 => VertexCostMap
                             , Context3 => Graph
                             , Iterator => EdgeIterator
                             , State => EdgeCostMap
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
                             , step => reweightEdgeLoopStep
                             , forLoopRepeat => reweightEdgeLoopRepeat
                             ];

    // Dijkstra loop
    use CppForIteratorLoop1_3[ Context1 => EdgeCostMap
                             , Context2 => VertexCostMap
                             , Context3 => Graph
                             , Iterator => VertexIterator
                             , State => VertexCostMatrix
                             , iterEnd => vertexIterEnd
                             , iterNext => vertexIterNext
                             , step => dijkstraAndAdjustLoopStep
                             , forLoopRepeat => dijkstraAndAdjustLoopRepeat
                             ];

    // Adjust loop
    use CppForIteratorLoop1_2[ Context1 => VertexDescriptor
                             , Context2 => VertexCostMap
                             , Iterator => VertexIterator
                             , State => VertexCostMap
                             , iterEnd => vertexIterEnd
                             , iterNext => vertexIterNext
                             , step => adjustVertexLoopStep
                             , forLoopRepeat => adjustVertexLoopRepeat
                             ];
}

// TODO: compiler bug below, investigate
// program CppJohnson = {
//     use GenericJohnson;

//     use CppDijkstraVisitor;
//     use CppBellmanFord;
// }