package examples.bgl.mg-src.bgl-py
    imports examples.bgl.mg-src.bfs_utils
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

    use PyForIteratorLoop3_2[ Iterator => EdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => VertexVector
                             , State2 => FIFOQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
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

    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph;

    use PyPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
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

    use PyForIteratorLoop3_2[ Iterator => EdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => VertexVector
                             , State2 => Stack
                             , State3 => ColorPropertyMap
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
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

    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph;

    use PyPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
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
    use PyForParallelIteratorLoop3_2[ Iterator => EdgeIterator
                                     , Context1 => Graph
                                     , Context2 => VertexDescriptor
                                     , State1 => VertexVector
                                     , State2 => FIFOQueue
                                     , State3 => ColorPropertyMap
                                     , iterEnd => edgeIterEnd
                                     // must be equivalent to incrementation
                                     , iterNext => edgeIterNext
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

    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph;

    use PyPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
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

    use PyForIteratorLoop3_2[ Iterator => EdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => StateWithMaps
                             , State2 => PriorityQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
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
                                           , KeyListIterator => EdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyMap => emptyECMap
                                           , iterEnd => edgeIterEnd
                                           , iterNext => edgeIterNext
                                           , iterUnpack => edgeIterUnpack
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

    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph;

    use PyPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
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

    use PyForIteratorLoop3_2[ Iterator => EdgeIterator
                             , Context1 => Graph
                             , Context2 => VertexDescriptor
                             , State1 => StateWithMaps
                             , State2 => PriorityQueue
                             , State3 => ColorPropertyMap
                             , iterEnd => edgeIterEnd
                             , iterNext => edgeIterNext
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
                                           , KeyListIterator => EdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyMap => emptyECMap
                                           , iterEnd => edgeIterEnd
                                           , iterNext => edgeIterNext
                                           , iterUnpack => edgeIterUnpack
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

    // PyIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use PyIncidenceAndVertexListGraph;

    use PyPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
               ];

    use PyVector[ A => VertexDescriptor
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};