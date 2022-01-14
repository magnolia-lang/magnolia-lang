// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/graph_theory_review.html#sec:bfs-algorithm
// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/breadth_first_search.html
package examples.bgl.mg-src.bfs
    imports examples.bgl.mg-src.color_marker
          , examples.bgl.mg-src.for_loop
          , examples.bgl.mg-src.graph
          , examples.bgl.mg-src.property_map
          , examples.bgl.mg-src.queue
          , examples.bgl.mg-src.stack
          , examples.bgl.mg-src.tuple
          , examples.bgl.mg-src.while_loop;


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
    require BFSVisitor[ Vertex => VertexDescriptor
                      , Edge => EdgeDescriptor
                      ];
    require Queue[ A => VertexDescriptor
                 , isEmpty => isEmptyQueue
                 ];
    require VertexListGraph;
    require IncidenceGraph;
    require ColorMarker;
    require ReadWritePropertyMap[ PropertyMap => ColorPropertyMap
                                , Key => VertexDescriptor
                                , Value => Color
                                ];

    require function initMap(vitr: VertexIterator,
                             c: Color): ColorPropertyMap;

    procedure breadthFirstVisit(obs g: Graph,
                                obs s: VertexDescriptor,
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

    use ForIteratorLoop3_2[ iterEnd => edgeIterEnd
                          , iterNext => edgeIterNext
                          , forLoopRepeat => bfsInnerLoopRepeat
                          , step => bfsInnerLoopStep
                          , Iterator => EdgeIterator
                          , State1 => A
                          , State2 => Queue
                          , State3 => ColorPropertyMap
                          , Context1 => Graph
                          , Context2 => VertexDescriptor
                          ];

    predicate bfsOuterLoopCond(a: A, q: Queue, c: ColorPropertyMap, g: Graph) {
        value !isEmptyQueue(q);
    }

    procedure bfsOuterLoopStep(upd x: A,
                               upd q: Queue,
                               upd c: ColorPropertyMap,
                               obs g: Graph) {
        var u = front(q);
        call pop(q);

        call examineVertex(u, g, q, x);

        var edgeItr: EdgeIterator;
        
        call outEdges(u, g, edgeItr);

        call bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);

        call put(c, u, black());
        call finishVertex(u, g, q, x);
    }

    type EdgeIterator;
    require procedure edgeIterNext(upd ei: EdgeIterator);
    require function edgeIterUnpack(ei: EdgeIterator): EdgeDescriptor;
    require function tgt(ed: EdgeDescriptor, g: Graph): VertexDescriptor;

    procedure bfsInnerLoopStep(obs edgeItr: EdgeIterator,
                               upd x: A,
                               upd q: Queue,
                               upd c: ColorPropertyMap,
                               obs g: Graph,
                               obs u: VertexDescriptor) {
        var e = edgeIterUnpack(edgeItr);

        var v = tgt(e, g);

        call examineEdge(e, g, q, x);

        var vc = get(c, v);

        if vc == white() then {
            call treeEdge(e, g, q, x);
            call put(c, v, gray());
            call discoverVertex(v, g, q, x);
            call push(v, q);

        } else if vc == gray() then {
            call grayTarget(e, g, q, x);
                                      
        } else { // vc == black();
            call blackTarget(e, g, q, x);
        };
    }
}

implementation BFSOrDFS = {
    use BFSVisit;
    require function empty(): Queue;
    
    function search(g: Graph,
                    start: VertexDescriptor,
                    init: A): A = {
        var q = empty(): Queue;
        var vertexItr: VertexIterator;
        call vertices(g, vertexItr);
        var c = initMap(vertexItr, white());
        var a = init;

        call breadthFirstVisit(g, start, a, q, c);
        value a;
    }
}

implementation BFS = {
    use BFSOrDFS[ search => breadthFirstSearch
                , Queue => FIFOQueue
                ];
    use FIFOQueue[ A => VertexDescriptor
                 , isEmpty => isEmptyQueue
                 ];
}

implementation DFS = {
    use BFSOrDFS[ search => depthFirstSearch
                , Queue => Stack // LIFOQueue
                , front => top
                , isEmptyQueue => isEmptyStack
                ];
    use Stack[ A => VertexDescriptor
             , isEmpty => isEmptyStack
             ];
}
