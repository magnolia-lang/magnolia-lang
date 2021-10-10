// TODO: call it bfs_test again, maybe
package examples.bgl_v2.mg-src.bfs_utils
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
        call cons(v, a);
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
