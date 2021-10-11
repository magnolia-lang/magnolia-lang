// TODO: call it bfs_test again, maybe
package examples.bgl_v2.mg-src.dfs_utils
    imports examples.bgl_v2.mg-src.bfs
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.list
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.stack
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.while_loop
          , examples.bgl_v2.mg-src.externals.cpp_apis;

implementation DFS = {
    use BFSVisit[ Queue => Stack
                , breadthFirstVisit => depthFirstVisit
                , front => top
                , isEmptyQueue => isEmptyStack
                ];
    use Stack[ A => Vertex
             , isEmpty => isEmptyStack
             , top => top
             ];

    function depthFirstSearch(g: Graph,
                              start: Vertex,
                              init: A): A = {
        var stack = empty(): Stack;
        var c = initMap(vertices(g), white());
        var a = init;

        call depthFirstVisit(g, start, a, stack, c);
        value a;
    }
}

implementation GenericDFSTestVisitor = {
    use DFS[ A => VertexList
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
                             upd q: Stack,
                             upd a: VertexList) = {
        a = cons(v, a);
    }

    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , Queue => Stack
                               , defaultAction => defaultAction
                               , A => VertexList
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , Queue => Stack
                               , defaultAction => defaultAction
                               , A => VertexList
                               ];
}
