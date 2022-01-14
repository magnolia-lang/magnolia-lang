// TODO: call it dfs_test again, maybe
package examples.bgl.mg-src.dfs_utils
    imports examples.bgl.mg-src.bfs
          , examples.bgl.mg-src.graph
          , examples.bgl.mg-src.list
          , examples.bgl.mg-src.property_map
          , examples.bgl.mg-src.queue
          , examples.bgl.mg-src.tuple
          , examples.bgl.mg-src.vector
          , examples.bgl.mg-src.while_loop
          , examples.bgl.mg-src.externals.cpp_apis;

// See Haskell example TestVisitor1 from the comparing generics paper
implementation GenericDFSTestVisitor = {
    use DFS[ A => VertexVector
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
                            upd q: Stack,
                            upd a: VertexVector) = {
        call pushBack(v, a);
    }

    use BFSVisitorDefaultAction[ EdgeOrVertex => VertexDescriptor
                               , Queue => Stack
                               , defaultAction => defaultAction
                               , A => VertexVector
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => EdgeDescriptor
                               , Queue => Stack
                               , defaultAction => defaultAction
                               , A => VertexVector
                               ];
}
