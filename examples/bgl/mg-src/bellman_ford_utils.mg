package examples.bgl.mg-src.bellman_ford_utils
    imports examples.bgl.mg-src.bool
          , examples.bgl.mg-src.for_loop
          // TODO: the following line triggers a compiler bug if removed and
          //       we write 'require VertexListGraph', for instance.
          , examples.bgl.mg-src.graph
          , examples.bgl.mg-src.list
          , examples.bgl.mg-src.property_map
          , examples.bgl.mg-src.relax
          , examples.bgl.mg-src.tuple
          , examples.bgl.mg-src.unit;

concept BellmanFordVisitor = {
  // Graph components
  type Graph;
  type Edge;

  // Return type
  type A;

  // BGL API for Bellman-Ford
  procedure examineEdge(obs e: Edge, obs g: Graph, upd a: A);
  procedure edgeRelaxed(obs e: Edge, obs g: Graph, upd a: A);
  procedure edgeNotRelaxed(obs e: Edge, obs g: Graph, upd a: A);
  procedure edgeMinimized(obs e: Edge, obs g: Graph, upd a: A);
  procedure edgeNotMinimized(obs e: Edge, obs g: Graph, upd a: A);
}

implementation BellmanFordVisitorDefaultAction = {
  type Graph;
  type Edge;
  type A;

  procedure defaultAction(obs e: Edge, obs g: Graph, upd a: A) {}
}

implementation BellmanFordBase = {
  // Graph basic utils
  require VertexListGraph;
  require IncidenceGraph;
  require EdgeListGraph;

  // Bool utils
  require Bool;
  // Unit utils
  require Unit;

  use Relax;

  use BellmanFordVisitor[ Edge => EdgeDescriptor
                        , Graph => Graph
                        , A => A
                        ];


  // Predicate interpretation of booleans
  predicate holds(b: Bool) = b == btrue();

  // Bellman-Ford shortest paths entry point.
  // As per the BGL documentation, the user must assign the source vertex a
  // cost of 0 and all other vertices a cost of infinity prior to calling
  // bellmanFordShortestPaths.
  procedure bellmanFordShortestPaths(obs g: Graph,
                                     upd vcm: VertexCostMap,
                                     obs ecm: EdgeCostMap,
                                     upd a: A,
                                     out vpm: VertexPredecessorMap,
                                     out allMinimized: Bool) {
    //var nbVertices = numVertices(g);

    var vertexItr: VertexIterator;
    call vertices(g, vertexItr);

    vpm = emptyVPMap();

    // Initialization of the predecessor map.
    // This is optional in the BGL implementation.
    // TODO: see if that should be moved out, for performance measurements.
    call populateVPMapLoopRepeat(vertexItr, vpm, unit());

    // Outer relaxation loop. Run once for each vertex.
    call edgeRelaxationOuterLoopRepeat(vertexItr, a, vcm, vpm, ecm, g);

    // Check for negative cycles
    var edgeItr: EdgeIterator;
    call edges(g, edgeItr);

    allMinimized = btrue();
    call checkNegativeCycleLoopRepeat(edgeItr, a, allMinimized, vcm, ecm, g);
  }

  // Edge Relaxation outer loop utils
  use ForIteratorLoop3_2[ Context1 => EdgeCostMap
                        , Context2 => Graph
                        , Iterator => VertexIterator
                        , State1 => A
                        , State2 => VertexCostMap
                        , State3 => VertexPredecessorMap
                        , iterEnd => vertexIterEnd
                        , iterNext => vertexIterNext
                        , step => edgeRelaxationOuterLoopStep
                        , forLoopRepeat => edgeRelaxationOuterLoopRepeat
                        ];

  // TODO: switch to breakable loop to optimize like BGL?
  // TODO: passing a vertexItr as the first parameter, but this is not
  // strictly necessary. The BGL body uses an int, but it is not actually used
  // anyway.
  procedure edgeRelaxationOuterLoopStep(obs vertexItr: VertexIterator,
                                        upd a: A,
                                        upd vcm: VertexCostMap,
                                        upd vpm: VertexPredecessorMap,
                                        obs ecm: EdgeCostMap,
                                        obs g: Graph) {
    var edgeItr: EdgeIterator;
    call edges(g, edgeItr);

    call edgeRelaxationInnerLoopRepeat(edgeItr, a, vcm, vpm, ecm, g);
  }

  // Edge relaxation inner loop utils
  use ForIteratorLoop3_2[ Context1 => EdgeCostMap
                        , Context2 => Graph
                        , Iterator => EdgeIterator
                        , State1 => A
                        , State2 => VertexCostMap
                        , State3 => VertexPredecessorMap
                        , iterEnd => edgeIterEnd
                        , iterNext => edgeIterNext
                        , step => edgeRelaxationInnerLoopStep
                        , forLoopRepeat => edgeRelaxationInnerLoopRepeat
                        ];
  // TODO: adding bool to relax could make it potentially much faster,
  // if need be.
  procedure edgeRelaxationInnerLoopStep(obs edgeItr: EdgeIterator,
                                        upd a: A,
                                        upd vcm: VertexCostMap,
                                        upd vpm: VertexPredecessorMap,
                                        obs ecm: EdgeCostMap,
                                        obs g: Graph) {
    var currentEdge = edgeIterUnpack(edgeItr);
    var origVcm = vcm;
    call relax(currentEdge, g, ecm, vcm, vpm);

    if vcm == origVcm
    then call edgeRelaxed(currentEdge, g, a)
    else call edgeNotRelaxed(currentEdge, g, a);
  }

  // Check negative cycle loop utils
  use ForIteratorLoop2_3[ Context1 => VertexCostMap
                        , Context2 => EdgeCostMap
                        , Context3 => Graph
                        , Iterator => EdgeIterator
                        , State1 => A
                        , State2 => Bool
                        , iterEnd => edgeIterEnd
                        , iterNext => edgeIterNext
                        , step => checkNegativeCycleLoopStep
                        , forLoopRepeat => checkNegativeCycleLoopRepeat
                        ];

  procedure checkNegativeCycleLoopStep(obs edgeItr: EdgeIterator,
                                       upd a: A,
                                       upd allMinimized: Bool,
                                       obs vcm: VertexCostMap,
                                       obs ecm: EdgeCostMap,
                                       obs g: Graph) {
    var currentEdge = edgeIterUnpack(edgeItr);

    var u = src(currentEdge, g);
    var v = tgt(currentEdge, g);

    var uCost = get(vcm, u);
    var vCost = get(vcm, v);

    var edgeCost = get(ecm, currentEdge);

    if less(plus(uCost, edgeCost), vCost) then {
      call edgeNotMinimized(currentEdge, g, a);
      allMinimized = bfalse();
    } else call edgeMinimized(currentEdge, g, a);
  }

  // VertexPredecessorMap utils
  use ForIteratorLoop[ Context => Unit
                     , Iterator => VertexIterator
                     , State => VertexPredecessorMap
                     , iterEnd => vertexIterEnd
                     , iterNext => vertexIterNext
                     , step => populateVPMapLoopStep
                     , forLoopRepeat => populateVPMapLoopRepeat
                     ];

  procedure populateVPMapLoopStep(obs itr: VertexIterator,
                                  upd vpm: VertexPredecessorMap,
                                  obs u: Unit) {
    var v = vertexIterUnpack(itr);
    call put(vpm, v, v);
  }

  require function emptyVPMap(): VertexPredecessorMap;
}

implementation GenericBellmanFord = {
  use BellmanFordBase;
  use BellmanFordVisitorDefaultAction[ Edge => EdgeDescriptor
                                     , defaultAction => edgeMinimized
                                     ];
  use BellmanFordVisitorDefaultAction[ Edge => EdgeDescriptor
                                     , defaultAction => edgeNotMinimized
                                     ];
  use BellmanFordVisitorDefaultAction[ Edge => EdgeDescriptor
                                     , defaultAction => edgeRelaxed
                                     ];
  use BellmanFordVisitorDefaultAction[ Edge => EdgeDescriptor
                                     , defaultAction => edgeNotRelaxed
                                     ];
  use BellmanFordVisitorDefaultAction[ Edge => EdgeDescriptor
                                     , defaultAction => examineEdge
                                     ];
}[ A => Unit ];