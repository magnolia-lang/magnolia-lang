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

implementation BFS = {
    require BFSVisitor;
    require VertexListGraph;
    require IncidenceGraph;
    require Queue[ A => Vertex ];
    require ColorMarker;
    require ReadWritePropertyMap[ PropertyMap => ColorPropertyMap
                                , Key => Vertex
                                , Value => Color
                                ];

    require function initMap(vl: VertexList, c: Color): ColorPropertyMap;
    require function empty(): Queue;

    function breadthFirstSearch(g: Graph,
                                start: Vertex,
                                init: A): A = {
        var q = empty(): Queue;
        var c = initMap(vertices(g), white());
        var a = init;

        call breadthFirstVisit(g, start, a, q, c);
        value a;
    }

    procedure breadthFirstVisit(obs g: Graph,
                                obs s: Vertex,
                                upd a: A,
                                upd q: Queue,
                                upd c: ColorPropertyMap) {
        call discoverVertex(s, g, q, a);

        var q1 = push(s, q);
        var c1 = put(c, s, gray());

        var outerState = makeOuterLoopState(a, q1, c1);
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

    use WhileLoop[ repeat => bfsInnerLoopRepeat
                 , step => bfsInnerLoopStep
                 , cond => bfsInnerLoopCond
                 , State => InnerLoopState
                 , Context => InnerLoopContext
                 ];

    use Pair[ A => Graph
            , B => Vertex
            , Pair => InnerLoopContext
            , makePair => makeInnerLoopContext
               // Renaming axiom avoids merging errors tagged as compiler bug.
               // Problem should be related to https://github.com/magnolia-lang/magnolia-lang/issues/43
               ];

    use Pair[ A => OuterLoopState
            , B => EdgeList
            , Pair => InnerLoopState
            , makePair => makeInnerLoopState
            ];

    require type InnerLoopContext;

    predicate bfsOuterLoopCond(state: OuterLoopState, g: Graph) {
        var q = second(state);
        value !isEmpty(q);
    }

    procedure bfsOuterLoopStep(upd state: OuterLoopState,
                               obs g: Graph) {
        var x = first(state);
        var q1 = second(state);
        var c = third(state);

        var u = front(q1);
        var q2 = pop(q1);

        call examineVertex(u, g, q2, x);

        var innerState = makeInnerLoopState(makeOuterLoopState(x, q2, c),
                                            outEdges(u, g));
        var innerContext = makeInnerLoopContext(g, u);

        call bfsInnerLoopRepeat(innerState, innerContext);

        var outerLoopStateAfterInnerLoop = first(innerState);
        var x_end = first(outerLoopStateAfterInnerLoop);
        var q_end = second(outerLoopStateAfterInnerLoop);
        var c_end = third(outerLoopStateAfterInnerLoop);

        call finishVertex(u, g, q_end, x_end);

        state = makeOuterLoopState(x_end, q_end, c_end);
    }

    require predicate isEmpty(el: EdgeList);
    require function head(el: EdgeList): Edge guard !isEmpty(el);
    require function tail(el: EdgeList): EdgeList guard !isEmpty(el);

    predicate bfsInnerLoopCond(state: InnerLoopState,
                               ctx: InnerLoopContext) {
        var edgeList = second(state);
        value !isEmpty(edgeList);
    }
    
    procedure bfsInnerLoopStep(upd state: InnerLoopState,
                               obs ctx: InnerLoopContext) {
        var g = first(ctx);
        var u = second(ctx);

        var outerState = first(state);
        var x1 = first(outerState);
        var q1 = second(outerState);
        var c1 = third(outerState);

        var edgeList = second(state);
        var e = head(edgeList);
        var es = tail(edgeList);

        var v = tgt(e);

        call examineEdge(e, g, q1, x1);

        var vc = get(c1, v);

        if vc == white() then {
            call treeEdge(e, g, q1, x1);
            var c2 = put(c1, v, gray());
            call discoverVertex(v, g, q1, x1);

            state = makeInnerLoopState(makeOuterLoopState(x1, push(v, q1), c2),
                                       es);
        } else if vc == gray() then {
            call grayTarget(e, g, q1, x1);
            state = makeInnerLoopState(makeOuterLoopState(x1, q1, c1), es);
                                      
        } else { // vc == black();
            call blackTarget(e, g, q1, x1);
            var c2 = put(c1, u, black());
            state = makeInnerLoopState(makeOuterLoopState(x1, q1, c2), es);
        };
    }
}
