package examples.bgl.mg-src.bgl-py
    imports examples.bgl.mg-src.bellman_ford_utils
          , examples.bgl.mg-src.bfs_utils
          , examples.bgl.mg-src.dfs_utils
          , examples.bgl.mg-src.dijkstra_utils
          , examples.bgl.mg-src.prim_utils
          , examples.bgl.mg-src.externals.python_apis;

program PyBFSTestVisitor = {
    use GenericBFSTestVisitor;

    use PyFIFOQueue[ A => VertexDescriptor
                    , isEmpty => isEmptyQueue
                    ];

    use PyWhileLoop3[ Context => Graph
                     , State1 => VertexVector
                     , State2 => FIFOQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use PyForIteratorLoop3_2[ Iterator => OutEdgeIterator
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


    use PyReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use PyBaseTypes;

    // PyIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListAndEdgeListGraph;

    use PyPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use PyVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};


program PyDFSTestVisitor = {
    use GenericDFSTestVisitor;

    use PyStack[ A => VertexDescriptor
                , isEmpty => isEmptyStack
                ];

    use PyWhileLoop3[ Context => Graph
                     , State1 => VertexVector
                     , State2 => Stack
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use PyForIteratorLoop3_2[ Iterator => OutEdgeIterator
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


    use PyReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use PyBaseTypes;

    // PyIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListAndEdgeListGraph;

    use PyPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use PyVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};

/*
program PyParallelBFSTestVisitor = {
    use GenericBFSTestVisitor;

    use PyThreadSafeFIFOQueue[ A => VertexDescriptor
                              , isEmpty => isEmptyQueue
                              ];

    use PyWhileLoop3[ Context => Graph
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
    use PyForParallelIteratorLoop3_2[ Iterator => OutEdgeIterator
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


    use PyReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];

    use PyBaseTypes;

    // PyIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListAndEdgeListGraph;

    use PyPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use PyThreadSafeVector[ A => VertexDescriptor
                           , Vector => VertexVector
                           , empty => emptyVertexVector
                           ];
};
*/


program PyDijkstraVisitor = {
    use GenericDijkstraVisitor;

    use PyBaseFloatOps[ Float => Cost ];

    use PyUpdateablePriorityQueue[ A => VertexDescriptor
                                  , Priority => Cost
                                  , PriorityMap => VertexCostMap
                                  , empty => emptyPriorityQueue
                                  , isEmpty => isEmptyQueue
                                  ];

    use PyWhileLoop3[ Context => Graph
                     , State1 => StateWithMaps
                     , State2 => PriorityQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                     ];

    use PyForIteratorLoop3_2[ Iterator => OutEdgeIterator
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

    use PyForIteratorLoop[ Context => VertexDescriptor
                          , Iterator => VertexIterator
                          , State => VertexPredecessorMap
                          , iterEnd => vertexIterEnd
                          , iterNext => vertexIterNext
                          , step => populateVPMapLoopStep
                          , forLoopRepeat => populateVPMapLoopRepeat
                          ];

    use PyReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                        , KeyListIterator => VertexIterator
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];


    use PyReadWritePropertyMapWithInitList[ Key => EdgeDescriptor
                                           , KeyListIterator => OutEdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyMap => emptyECMap
                                           , iterEnd => outEdgeIterEnd
                                           , iterNext => outEdgeIterNext
                                           , iterUnpack => outEdgeIterUnpack
                                           ];

    use PyReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => VertexDescriptor
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyMap => emptyVPMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    use PyReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                           , KeyListIterator => VertexIterator
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyMap => emptyVCMap
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
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

    use PyBaseTypes;

    // PyIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListAndEdgeListGraph;

    use PyPair[ A => OutEdgeIterator
               , B => OutEdgeIterator
               , Pair => OutEdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeOutEdgeIteratorRange
               ];

    use PyVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};


program PyPrimVisitor = {
    use GenericPrimVisitor;

    use PyBaseFloatOps[ Float => Cost ];

    use PyUpdateablePriorityQueue[ A => VertexDescriptor
                                 , Priority => Cost
                                 , PriorityMap => VertexCostMap
                                 , empty => emptyPriorityQueue
                                 , isEmpty => isEmptyQueue
                                 ];

    use PyWhileLoop3[ Context => Graph
                    , State1 => StateWithMaps
                    , State2 => PriorityQueue
                    , State3 => ColorPropertyMap
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];

    use PyForIteratorLoop3_2[ Iterator => OutEdgeIterator
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

    use PyForIteratorLoop[ Context => VertexDescriptor
                         , Iterator => VertexIterator
                         , State => VertexPredecessorMap
                         , iterEnd => vertexIterEnd
                         , iterNext => vertexIterNext
                         , step => populateVPMapLoopStep
                         , forLoopRepeat => populateVPMapLoopRepeat
                         ];

    use PyReadWriteColorMapWithInitList[ Key => VertexDescriptor
                                       , KeyListIterator => VertexIterator
                                       , iterEnd => vertexIterEnd
                                       , iterNext => vertexIterNext
                                       , iterUnpack => vertexIterUnpack
                                       ];


    use PyReadWritePropertyMapWithInitList[ Key => EdgeDescriptor
                                          , KeyListIterator => OutEdgeIterator
                                          , Value => Cost
                                          , PropertyMap => EdgeCostMap
                                          , emptyMap => emptyECMap
                                          , iterEnd => outEdgeIterEnd
                                          , iterNext => outEdgeIterNext
                                          , iterUnpack => outEdgeIterUnpack
                                          ];

    use PyReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                          , KeyListIterator => VertexIterator
                                          , Value => VertexDescriptor
                                          , PropertyMap => VertexPredecessorMap
                                          , emptyMap => emptyVPMap
                                          , iterEnd => vertexIterEnd
                                          , iterNext => vertexIterNext
                                          , iterUnpack => vertexIterUnpack
                                          ];

    use PyReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                          , KeyListIterator => VertexIterator
                                          , Value => Cost
                                          , PropertyMap => VertexCostMap
                                          , emptyMap => emptyVCMap
                                          , iterEnd => vertexIterEnd
                                          , iterNext => vertexIterNext
                                          , iterUnpack => vertexIterUnpack
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

    use PyBaseTypes;

    // PyIncidenceAndVertexListAndEdgeListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListAndEdgeListGraph;

    use PyPair[ A => OutEdgeIterator
              , B => OutEdgeIterator
              , Pair => OutEdgeIteratorRange
              , first => iterRangeBegin
              , second => iterRangeEnd
              , makePair => makeOutEdgeIteratorRange
              ];

    use PyVector[ A => VertexDescriptor
                , Vector => VertexVector
                , empty => emptyVertexVector
                ];
};

program PyBellmanFord = {
    use GenericBellmanFord;

    use PyBaseFloatOps[ Float => Cost ];

    use PyForIteratorLoop[ Context => Unit
                         , Iterator => VertexIterator
                         , State => VertexPredecessorMap
                         , iterEnd => vertexIterEnd
                         , iterNext => vertexIterNext
                         , step => populateVPMapLoopStep
                         , forLoopRepeat => populateVPMapLoopRepeat
                         ];

    use PyForIteratorLoop3_2[ Context1 => EdgeCostMap
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

    use PyForIteratorLoop3_2[ Context1 => EdgeCostMap
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

    use PyForIteratorLoop2_3[ Context1 => VertexCostMap
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

    use PyReadWritePropertyMapWithInitList[ Key => EdgeDescriptor
                                          , KeyListIterator => OutEdgeIterator
                                          , Value => Cost
                                          , PropertyMap => EdgeCostMap
                                          , emptyMap => emptyECMap
                                          , iterEnd => outEdgeIterEnd
                                          , iterNext => outEdgeIterNext
                                          , iterUnpack => outEdgeIterUnpack
                                          ];

    use PyReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                          , KeyListIterator => VertexIterator
                                          , Value => VertexDescriptor
                                          , PropertyMap => VertexPredecessorMap
                                          , emptyMap => emptyVPMap
                                          , iterEnd => vertexIterEnd
                                          , iterNext => vertexIterNext
                                          , iterUnpack => vertexIterUnpack
                                          ];

    use PyReadWritePropertyMapWithInitList[ Key => VertexDescriptor
                                          , KeyListIterator => VertexIterator
                                          , Value => Cost
                                          , PropertyMap => VertexCostMap
                                          , emptyMap => emptyVCMap
                                          , iterEnd => vertexIterEnd
                                          , iterNext => vertexIterNext
                                          , iterUnpack => vertexIterUnpack
                                          ];

    use PyBaseTypes;

    use PyIncidenceAndVertexListAndEdgeListGraph;

    use PyBool;
    use PyUnit;
}