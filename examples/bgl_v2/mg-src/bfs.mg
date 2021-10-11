// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/graph_theory_review.html#sec:bfs-algorithm
// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/breadth_first_search.html
package examples.bgl_v2.mg-src.bfs
    imports examples.bgl_v2.mg-src.color_marker
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.while_loop;


concept BFSVisitor = {
    
    // Graph components
    type Graph;
    type Edge;
    type Vertex;

    type Queue;

    // Return type
    type A;

    // BFS BGL API
    // TODO: add guards for Vertex having to be in Graph?
    procedure discoverVertex(obs v: Vertex,
                             obs g: Graph,
                             upd q: Queue,
                             upd a: A);

    procedure examineVertex(obs v: Vertex,
                            obs g: Graph,
                            upd q: Queue,
                            upd a: A);

    procedure examineEdge(obs e: Edge,
                          obs g: Graph,
                          upd q: Queue,
                          upd a: A);

    procedure treeEdge(obs e: Edge,
                       obs g: Graph,
                       upd q: Queue,
                       upd a: A);
    
    procedure nonTreeEdge(obs e: Edge,
                          obs g: Graph,
                          upd q: Queue,
                          upd a: A);

    procedure grayTarget(obs e: Edge,
                         obs g: Graph,
                         upd q: Queue,
                         upd a: A);
    
    procedure blackTarget(obs e: Edge,
                          obs g: Graph,
                          upd q: Queue,
                          upd a: A);

    procedure finishVertex(obs v: Vertex,
                           obs g: Graph,
                           upd q: Queue,
                           upd a: A);
}

// Use BFSVisitorDefaultAction with appropriate renamings to provide default
// implementations to the procedures declared in BFSVisitor.
implementation BFSVisitorDefaultAction = {
    type A;
    type EdgeOrVertex;
    type Graph;
    type Queue;

    procedure defaultAction(obs edgeOrVertex: EdgeOrVertex,
                            obs g: Graph,
                            upd q: Queue,
                            upd a: A) = {};
}

implementation BFSVisit = {
    require BFSVisitor;
    require Queue[ A => Vertex
                 , isEmpty => isEmptyQueue
                 ];
    require VertexListGraph;
    require IncidenceGraph;
    require ColorMarker;
    require ReadWritePropertyMap[ PropertyMap => ColorPropertyMap
                                , Key => Vertex
                                , Value => Color
                                ];

    require function initMap(vl: VertexList, c: Color): ColorPropertyMap;

    procedure breadthFirstVisit(obs g: Graph,
                                obs s: Vertex,
                                upd a: A,
                                upd q: Queue,
                                upd c: ColorPropertyMap) {
        call discoverVertex(s, g, q, a);

        call push(s, q);
        call put(c, s, gray());

        call bfsOuterLoopRepeat(a, q, c, g);
    }

    use WhileLoop3[ repeat => bfsOuterLoopRepeat
                 , step => bfsOuterLoopStep
                 , cond => bfsOuterLoopCond
                 , State1 => A
                 , State2 => Queue
                 , State3 => ColorPropertyMap
                 , Context => Graph
                 ];

    use WhileLoop4_3[ repeat => bfsInnerLoopRepeat
                    , step => bfsInnerLoopStep
                    , cond => bfsInnerLoopCond
                    , State1 => A
                    , State2 => Queue
                    , State3 => ColorPropertyMap
                    , State4 => EdgeIterator
                    , Context1 => Graph
                    , Context2 => Vertex
                    , Context3 => EdgeIterator
                    ];

    predicate bfsOuterLoopCond(a: A, q: Queue, c: ColorPropertyMap, g: Graph) {
        value !isEmptyQueue(q);
    }

    procedure bfsOuterLoopStep(upd x: A, upd q: Queue, upd c: ColorPropertyMap,
                               obs g: Graph) {
        var u = front(q);
        call pop(q);

        call examineVertex(u, g, q, x);

        var edgeItrBegin: EdgeIterator;
        var edgeItrEnd: EdgeIterator;
        
        call outEdges(u, g, edgeItrBegin, edgeItrEnd);

        call bfsInnerLoopRepeat(x, q, c, edgeItrBegin, g, u, edgeItrEnd);

        call finishVertex(u, g, q, x);
    }

    type EdgeIterator;
    require procedure iterNext(upd ei: EdgeIterator);
    require function iterUnpack(ei: EdgeIterator): Edge;

    predicate bfsInnerLoopCond(x: A,
                               q: Queue,
                               c: ColorPropertyMap,
                               edgeItr: EdgeIterator,
                               g: Graph,
                               u: Vertex,
                               edgeItrEnd: EdgeIterator) {
        value edgeItr != edgeItrEnd;
    }
    
    procedure bfsInnerLoopStep(upd x1: A, upd q1: Queue,
                               upd c: ColorPropertyMap,
                               upd edgeItr: EdgeIterator,
                               obs g: Graph,
                               obs u: Vertex,
                               obs edgeItrEnd: EdgeIterator) {
        var e = iterUnpack(edgeItr);
        call iterNext(edgeItr);

        var v = tgt(e);

        call examineEdge(e, g, q1, x1);

        var vc = get(c, v);

        if vc == white() then {
            call treeEdge(e, g, q1, x1);
            call put(c, v, gray());
            call discoverVertex(v, g, q1, x1);
            call push(v, q1);

        } else if vc == gray() then {
            call grayTarget(e, g, q1, x1);
                                      
        } else { // vc == black();
            call blackTarget(e, g, q1, x1);
            call put(c, u, black());
        };
    }
}

implementation BFS = {
    use BFSVisit[ Queue => FIFOQueue ];
    use FIFOQueue[ A => Vertex
                 , isEmpty => isEmptyQueue
                 ];
    function breadthFirstSearch(g: Graph,
                                start: Vertex,
                                init: A): A = {
        var q = empty(): FIFOQueue;
        var c = initMap(vertices(g), white());
        var a = init;

        call breadthFirstVisit(g, start, a, q, c);
        value a;
    }
}
