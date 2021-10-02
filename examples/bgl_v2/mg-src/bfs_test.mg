package examples.bgl.mg-src.v2.bfs_test
    imports examples.bgl.mg-src.v2.bfs
          , examples.bgl.mg-src.v2.graph
          , examples.bgl.mg-src.v2.list
          , examples.bgl.mg-src.v2.property_map
          , examples.bgl.mg-src.v2.queue
          , examples.bgl.mg-src.v2.tuple
          , examples.bgl.mg-src.v2.while_loop
          , examples.bgl.mg-src.v2.externals.cpp.base_types_cpp
          , examples.bgl.mg-src.v2.externals.cpp.color_marker_cpp
          , examples.bgl.mg-src.v2.externals.cpp.graph_cpp
          , examples.bgl.mg-src.v2.externals.cpp.list_cpp
          , examples.bgl.mg-src.v2.externals.cpp.property_map_cpp
          , examples.bgl.mg-src.v2.externals.cpp.queue_cpp
          , examples.bgl.mg-src.v2.externals.cpp.tuple_cpp
          , examples.bgl.mg-src.v2.externals.cpp.while_loop_cpp;

// See Haskell example TestVisitor1 from the comparing generics paper
program BFSTestVisitor = {
    use BFS[ A => IntList ];

    procedure discoverVertex(obs v: Vertex,
                             obs g: Graph,
                             upd q: Queue,
                             upd a: IntList) = { // A should be list, perhaps?
        // TODO 
    }

    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , defaultAction => examineVertex
                               , A => IntList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => examineEdge
                               , A => IntList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => treeEdge
                               , A => IntList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => nonTreeEdge
                               , A => IntList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => grayTarget
                               , A => IntList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => blackTarget
                               , A => IntList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , defaultAction => finishVertex
                               , A => IntList
                               ];

    use CppColorMarker;
    use CppList[ A => Edge
               , List => EdgeList
               ];
    use CppList[ A => Vertex
               , List => VertexList
               ];
    use CppList[ A => Int
               , List => IntList
               ];

    use CppTriplet[ A => IntList
                  , B => Queue
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

    use CppQueue[ A => Vertex ];
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
                                           ];

    use CppBaseTypes;
    use CppEdge;
    use CppGraphType;
    use CppIncidenceGraph;
    use CppVertexListGraph;
};
