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

        var outerState = makeOuterLoopState(a, q, c);
        call bfsOuterLoopRepeat(outerState, g);
        a = first(outerState);
    }

    use WhileLoop[ repeat => bfsOuterLoopRepeat
                 , step => bfsOuterLoopStep
                 , cond => bfsOuterLoopCond
                 , State => OuterLoopState
                 , Context => Graph
                 ];

    use Triplet[ A => A
               , B => Queue
               , C => ColorPropertyMap
               , Triplet => OuterLoopState
               , makeTriplet => makeOuterLoopState
               ];

    use WhileLoop4[ repeat => bfsInnerLoopRepeat
                  , step => bfsInnerLoopStep
                  , cond => bfsInnerLoopCond
                  , State1 => A
                  , State2 => Queue
                  , State3 => ColorPropertyMap
                  , State4 => EdgeList
                  , Context => InnerLoopContext
                 ];

    use Pair[ A => Graph
            , B => Vertex
            , Pair => InnerLoopContext
            , makePair => makeInnerLoopContext
               // Renaming axiom avoids merging errors tagged as compiler bug.
               // Problem should be related to https://github.com/magnolia-lang/magnolia-lang/issues/43
               ];

    require type InnerLoopContext;

    predicate bfsOuterLoopCond(state: OuterLoopState, g: Graph) {
        var q = second(state);
        value !isEmptyQueue(q);
    }

    procedure bfsOuterLoopStep(upd state: OuterLoopState,
                               obs g: Graph) {
        var x = first(state);
        var q = second(state);
        var c = third(state);

        var u = front(q);
        call pop(q);

        call examineVertex(u, g, q, x);

        var edges = outEdges(u, g);
        var innerContext = makeInnerLoopContext(g, u);

        call bfsInnerLoopRepeat(x, q, c, edges, innerContext);

        call finishVertex(u, g, q, x);

        state = makeOuterLoopState(x, q, c);
    }

    require predicate isEmpty(el: EdgeList);
    require function head(el: EdgeList): Edge guard !isEmpty(el);
    require procedure tail(upd el: EdgeList) guard !isEmpty(el);

    predicate bfsInnerLoopCond(x: A, q: Queue, c: ColorPropertyMap,
                               edgeList: EdgeList,
                               ctx: InnerLoopContext) {
        value !isEmpty(edgeList);
    }
    
    procedure bfsInnerLoopStep(upd x1: A, upd q1: Queue,
                               upd c: ColorPropertyMap,
                               upd edgeList: EdgeList,
                               obs ctx: InnerLoopContext) {
        var g = first(ctx);
        var u = second(ctx);

        var e = head(edgeList);

        call tail(edgeList);

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
