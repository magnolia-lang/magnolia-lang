// TODO: call it bfs_test again, maybe
package examples.bgl_v2.mg-src.bfs_utils
    imports examples.bgl_v2.mg-src.bfs
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.list
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.vector
          , examples.bgl_v2.mg-src.while_loop
          , examples.bgl_v2.mg-src.externals.cpp_apis;

// See Haskell example TestVisitor1 from the comparing generics paper
implementation GenericBFSTestVisitor = {
    use BFS[ A => VertexVector
           , discoverVertex => defaultAction
           , examineEdge => defaultAction
           , treeEdge => defaultAction
           , nonTreeEdge => defaultAction
           , grayTarget => defaultAction
           , blackTarget => defaultAction
           , finishVertex => defaultAction
           ];

    use Vector[ A => VertexDescriptor
              , Vector => VertexVector
              , empty => emptyVertexVector
              ];

    procedure examineVertex(obs v: VertexDescriptor,
                            obs g: Graph,
                            upd q: FIFOQueue,
                            upd a: VertexVector) = {
        call pushBack(v, a);
    }

    use BFSVisitorDefaultAction[ EdgeOrVertex => VertexDescriptor
                               , Queue => FIFOQueue
                               , defaultAction => defaultAction
                               , A => VertexVector
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => EdgeDescriptor
                               , Queue => FIFOQueue
                               , defaultAction => defaultAction
                               , A => VertexVector
                               ];
}
