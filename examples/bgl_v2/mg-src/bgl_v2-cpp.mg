package examples.bgl_v2.mg-src.bgl_v2-cpp
    imports examples.bgl_v2.mg-src.bfs_utils
          , examples.bgl_v2.mg-src.dijkstra_utils
          , examples.bgl_v2.mg-src.externals.cpp_apis;

program CppBFSTestVisitor = {
    use GenericBFSTestVisitor;

    //use CppColorMarker;
    use CppIterableList[ A => Edge
                       , List => EdgeList
                       , ListIterator => EdgeIterator
                       , empty => emptyEdgeList
                       , iterBegin => edgeIterBegin
                       , iterEnd => edgeIterEnd
                       , iterNext => edgeIterNext
                       , iterUnpack => edgeIterUnpack
                       ];
    use CppIterableList[ A => Vertex
                       , List => VertexList
                       , ListIterator => VertexIterator
                       , empty => emptyVertexList
                       , iterBegin => vertexIterBegin
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

    use CppWhileLoop4_3[ Context1 => Graph 
                       , Context2 => Vertex
                       , Context3 => EdgeIterator
                       , State1 => VertexVector
                       , State2 => FIFOQueue
                       , State3 => ColorPropertyMap
                       , State4 => EdgeIterator
                       , cond => bfsInnerLoopCond
                       , step => bfsInnerLoopStep
                       , repeat => bfsInnerLoopRepeat
                       ];

    
    use CppReadWriteColorMapWithInitList[ Key => Vertex
                                        , KeyList => VertexList
                                        , KeyListIterator => VertexIterator
                                        , emptyKeyList => emptyVertexList
                                        , iterBegin => vertexIterBegin
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];
    
    /*use CppColorMarker;
    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , KeyListIterator => VertexIterator
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];
    */
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
    use CppPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
               ];

    use CppVector[ A => Vertex
                 , Vector => VertexVector
                 , empty => emptyVertexVector
                 ];
};

program CppDijkstraVisitor = {
    use GenericDijkstraVisitor;

    use CppIterableList[ A => Edge
                       , List => EdgeList
                       , ListIterator => EdgeIterator
                       , empty => emptyEdgeList
                       , iterBegin => edgeIterBegin
                       , iterEnd => edgeIterEnd
                       , iterNext => edgeIterNext
                       , iterUnpack => edgeIterUnpack
                       ];
    use CppIterableList[ A => Vertex
                       , List => VertexList
                       , ListIterator => VertexIterator
                       , empty => emptyVertexList
                       , iterBegin => vertexIterBegin
                       , iterEnd => vertexIterEnd
                       , iterNext => vertexIterNext
                       , iterUnpack => vertexIterUnpack
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
    use CppWhileLoop3[ Context => Graph
                     , State1 => StateWithMaps
                     , State2 => PriorityQueue
                     , State3 => ColorPropertyMap
                     , cond => bfsOuterLoopCond
                     , step => bfsOuterLoopStep
                     , repeat => bfsOuterLoopRepeat
                    ];
    
    use CppWhileLoop4_3[ Context1 => Graph
                       , Context2 => Vertex
                       , Context3 => EdgeIterator
                       , State1 => StateWithMaps
                       , State2 => PriorityQueue
                       , State3 => ColorPropertyMap
                       , State4 => EdgeIterator
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

    use CppReadWriteColorMapWithInitList[ Key => Vertex
                                        , KeyList => VertexList
                                        , KeyListIterator => VertexIterator
                                        , emptyKeyList => emptyVertexList
                                        , iterBegin => vertexIterBegin
                                        , iterEnd => vertexIterEnd
                                        , iterNext => vertexIterNext
                                        , iterUnpack => vertexIterUnpack
                                        ];


    /*use CppColorMarker;
    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , KeyListIterator => VertexIterator
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];

    */

    use CppReadWritePropertyMapWithInitList[ Key => Edge
                                           , KeyList => EdgeList
                                           , KeyListIterator => EdgeIterator
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyKeyList => emptyEdgeList
                                           , emptyMap => emptyECMap
                                           , iterBegin => edgeIterBegin
                                           , iterEnd => edgeIterEnd
                                           , iterNext => edgeIterNext
                                           , iterUnpack => edgeIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , KeyListIterator => VertexIterator
                                           , Value => Vertex
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVPMap
                                           , iterBegin => vertexIterBegin
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , KeyListIterator => VertexIterator
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVCMap
                                           , iterBegin => vertexIterBegin
                                           , iterEnd => vertexIterEnd
                                           , iterNext => vertexIterNext
                                           , iterUnpack => vertexIterUnpack
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
    use CppPair[ A => EdgeIterator
               , B => EdgeIterator
               , Pair => EdgeIteratorRange
               , first => iterRangeBegin
               , second => iterRangeEnd
               , makePair => makeEdgeIteratorRange
               ];
}
