// TODO: call it bfs_test again, maybe
package examples.bgl_v2.mg-src.bgl_v2
    imports examples.bgl_v2.mg-src.bfs
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.list
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.while_loop
          , examples.bgl_v2.mg-src.externals.cpp_apis;

// See Haskell example TestVisitor1 from the comparing generics paper
implementation GenericBFSTestVisitor = {
    use BFS[ A => VertexList
           , examineEdge => defaultAction
           , examineVertex => defaultAction
           , treeEdge => defaultAction
           , nonTreeEdge => defaultAction
           , grayTarget => defaultAction
           , blackTarget => defaultAction
           , finishVertex => defaultAction
           ];

    procedure discoverVertex(obs v: Vertex,
                             obs g: Graph,
                             upd q: FIFOQueue,
                             upd a: VertexList) = { // A should be list, perhaps?
        a = cons(v, a);
    }

    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , Queue => FIFOQueue
                               , defaultAction => defaultAction
                               , A => VertexList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , Queue => FIFOQueue
                               , defaultAction => defaultAction
                               , A => VertexList
                               ];
}

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

    use CppQueue[ A => Vertex
                , Queue => FIFOQueue
                ];
    use CppWhileLoop[ Context => Graph
                    , State => OuterLoopState
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];

    use CppWhileLoop[ Context => InnerLoopContext
                    , State => InnerLoopState
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
