// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/graph_theory_review.html#sec:bfs-algorithm
// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/breadth_first_search.html
package examples.bgl_v2.mg-src.bfs
    imports examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.property_map
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.while_loop;


concept ColorMarker = {
    type Color;

    function white(): Color;
    function gray(): Color;
    function black(): Color;

    axiom threeDistinctColors() {
        assert white() != gray();
        assert gray() != black();
        assert black() != white();
    }

    axiom exactlyThreeColors(c: Color) {
        assert c == white() || c == gray() || c == black();
    }
}

concept BFSVisitor = {
    type BFSVisitor;
    
    // Graph components
    type Graph;
    type Edge;
    type Vertex;

    type Queue;

    // Return type
    type A;

    // BFS BGL API
    // TODO: add guards for Vertex having to be in Graph?
    procedure discoverVertex(obs vis: BFSVisitor,
                             obs v: Vertex,
                             obs g: Graph,
                             upd q: Queue,
                             upd a: A);

    procedure examineVertex(obs vis: BFSVisitor,
                            obs v: Vertex,
                            obs g: Graph,
                            upd q: Queue,
                            upd a: A);

    procedure examineEdge(obs vis: BFSVisitor,
                          obs e: Edge,
                          obs g: Graph,
                          upd q: Queue,
                          upd a: A);

    procedure treeEdge(obs vis: BFSVisitor,
                       obs e: Edge,
                       obs g: Graph,
                       upd q: Queue,
                       upd a: A);
    
    procedure nonTreeEdge(obs vis: BFSVisitor,
                          obs e: Edge,
                          obs g: Graph,
                          upd q: Queue,
                          upd a: A);

    procedure grayTarget(obs vis: BFSVisitor,
                         obs e: Edge,
                         obs g: Graph,
                         upd q: Queue,
                         upd a: A);
    
    procedure blackTarget(obs vis: BFSVisitor,
                          obs e: Edge,
                          obs g: Graph,
                          upd q: Queue,
                          upd a: A);

    procedure finishVertex(obs vis: BFSVisitor,
                           obs v: Vertex,
                           obs g: Graph,
                           upd q: Queue,
                           upd a: A);
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
                                vis: BFSVisitor,
                                init: A): A = {
        var q = empty();
        var c = initMap(vertices(g), white());
        var a = init;

        call breadthFirstVisit(g, start, vis, a, q, c);
        value a;
    }

    procedure breadthFirstVisit(obs g: Graph,
                                obs v: Vertex,
                                obs vis: BFSVisitor,
                                upd a: A,
                                upd q: Queue,
                                upd c: ColorPropertyMap) {
        call discoverVertex(vis, v, g, q, a);

        var q1 = push(v, q);
        var c1 = put(c, v, gray());

        // TODO: as proc
        //a = bfsWhileLoop(q1, c1);
    }

    use WhileLoop[ repeat => bfsOuterLoopRepeat
                 , step => bfsOuterLoopStep
                 , cond => bfsOuterLoopCond
                 , State => OuterLoopState
                 , Context => OuterLoopContext
                 ];

    use Triplet[ A => A
               , B => Queue
               , C => ColorPropertyMap
               , Triplet => OuterLoopState
               , makeTriplet => makeOuterLoopState
               ];

    use Pair[ A => Graph
            , B => BFSVisitor
            , Pair => OuterLoopContext
            , makePair => makeOuterLoopContext
            ];

    use WhileLoop[ repeat => bfsInnerLoopRepeat
                 , step => bfsInnerLoopStep
                 , cond => bfsInnerLoopCond
                 , State => InnerLoopState
                 , Context => InnerLoopContext
                 ];

    use Triplet[ A => Graph
               , B => BFSVisitor
               , C => Vertex
               , Triplet => InnerLoopContext
               , makeTriplet => makeInnerLoopContext
               // Renaming axiom avoids merging errors tagged as compiler bug.
               // Problem should be related to https://github.com/magnolia-lang/magnolia-lang/issues/43
               ];

    use Pair[ A => OuterLoopState
            , B => EdgeList
            , Pair => InnerLoopState
            , makePair => makeInnerLoopState
            ];

    require type BFSVisitor; // TODO == OuterLoopContext
    require type InnerLoopContext;

    predicate bfsOuterLoopCond(state: OuterLoopState, vis: BFSVisitor) {
        var q = second(state);
        value isEmpty(q);
    }

    procedure bfsOuterLoopStep(upd state: OuterLoopState,
                               obs ctx: OuterLoopContext) {
        var x = first(state);
        var q1 = second(state);
        var c = third(state);

        var u = front(q1);
        var q2 = pop(q1);

        var g = first(ctx);
        var vis = second(ctx);

        call examineVertex(vis, u, g, q2, x);

        var innerState = makeInnerLoopState(makeOuterLoopState(x, q2, c),
                                            outEdges(u, g));
        var innerContext = makeInnerLoopContext(g, vis, u);

        call bfsInnerLoopRepeat(innerState, innerContext);

        var outerLoopStateAfterInnerLoop = first(innerState);
        var x_end = first(outerLoopStateAfterInnerLoop);
        var q_end = second(outerLoopStateAfterInnerLoop);
        var c_end = third(outerLoopStateAfterInnerLoop);

        call finishVertex(vis, u, g, q_end, x_end);

        state = makeOuterLoopState(x_end, q_end, c_end);
    }

    require predicate isEmpty(el: EdgeList);
    require function head(el: EdgeList): Edge guard !isEmpty(el);
    require function tail(el: EdgeList): EdgeList guard !isEmpty(el);

    predicate bfsInnerLoopCond(state: InnerLoopState,
                               ctx: InnerLoopContext) {
        var edgeList = second(state);
        value isEmpty(edgeList);
    }
    
    procedure bfsInnerLoopStep(upd state: InnerLoopState,
                               obs ctx: InnerLoopContext) {
        var g = first(ctx);
        var vis = second(ctx);
        var u = third(ctx);

        var outerState = first(state);
        var x1 = first(outerState);
        var q1 = second(outerState);
        var c1 = third(outerState);

        var edgeList = second(state);
        var e = head(edgeList);
        var es = tail(edgeList);

        var v = tgt(e);

        call examineEdge(vis, e, g, q1, x1);

        var vc = get(c1, v);

        if vc == white() then {
            call treeEdge(vis, e, g, q1, x1);
            var c2 = put(c1, v, gray());
            call discoverVertex(vis, v, g, q1, x1);

            state = makeInnerLoopState(makeOuterLoopState(x1, push(v, q1), c2),
                                       es);
        } else if vc == gray() then {
            call grayTarget(vis, e, g, q1, x1);
            state = makeInnerLoopState(makeOuterLoopState(x1, q1, c1), es);
                                      
        } else { // vc == black();
            call blackTarget(vis, e, g, q1, x1);
            var c2 = put(c1, u, black());
            state = makeInnerLoopState(makeOuterLoopState(x1, q1, c1), es);
        };
    }
}
