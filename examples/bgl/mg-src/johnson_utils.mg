package examples.bgl.mg-src.johnson_utils
    imports examples.bgl.mg-src.bellman_ford_utils
          , examples.bgl.mg-src.dijkstra_utils
          , examples.bgl.mg-src.for_loop
          , examples.bgl.mg-src.property_map;

concept JohnsonVisitor = BellmanFordVisitor;

implementation JohnsonBase = {
  use BellmanFordBase;
  use GenericDijkstraVisitor[ EdgeCostMap => WriteableEdgeCostMap ];

  require function emptyVCMap(): VertexCostMap;
  require function emptyWECMap(): WriteableEdgeCostMap;
  require function zero(): Cost;

  use ReadWritePropertyMap[ Key => EdgeDescriptor
                          , Value => Cost
                          , PropertyMap => WriteableEdgeCostMap
                          ];

  use ReadWritePropertyMap[ Key => VertexDescriptor
                          , Value => VertexCostMap
                          , PropertyMap => VertexCostMatrix
                          ];

  procedure johnsonAllPairsShortestPaths(obs g: Graph,
                                         obs ecm: EdgeCostMap,
                                         upd a: A,
                                         upd vcmat: VertexCostMatrix,
                                         out succeeded: Bool) {
    var vertexItr: VertexIterator;
    call vertices(g, vertexItr);

    // Initialize vertex cost map
    var vcm = emptyVCMap();
    call initializeVertexCostMapLoopRepeat(vertexItr, vcm, zero());

    // Create out vertex predecessor map for Bellman-Ford algorithm
    var vpm: VertexPredecessorMap;

    // Get reweighting factors from a call to Bellman-Ford
    call bellmanFordShortestPaths(g, vcm, ecm, a, vpm, succeeded);

    // Abort if Bellman-Ford didn't succeed, i.e. if a negative cycle
    // was detected.
    if holds(succeeded) then {
      var edgeItr: EdgeIterator;
      call edges(g, edgeItr);

      // Reweighting
      var newCosts = emptyWECMap();
      call reweightEdgeLoopRepeat(edgeItr, newCosts, ecm, vcm, g);

      // Dijkstra call for each vertex + reweighting
      call dijkstraAndAdjustLoopRepeat(vertexItr, vcmat, newCosts, vcm, g);
    }
    else skip; // A negative cycle was detected, abort.
  }

  // Vertex cost initialization loop utils
  require ForIteratorLoop[ Context => Cost
                         , Iterator => VertexIterator
                         , State => VertexCostMap
                         , iterEnd => vertexIterEnd
                         , iterNext => vertexIterNext
                         , step => initializeVertexCostMapLoopStep
                         , forLoopRepeat => initializeVertexCostMapLoopRepeat
                         ];

  procedure initializeVertexCostMapLoopStep(obs vertexItr: VertexIterator,
                                            upd vcm: VertexCostMap,
                                            obs initialCost: Cost) {
    var currentVertex = vertexIterUnpack(vertexItr);
    call put(vcm, currentVertex, initialCost);
  }

  // Reweighting utils
  require function negate(c: Cost): Cost;

  require ForIteratorLoop1_3[ Context1 => EdgeCostMap
                            , Context2 => VertexCostMap
                            , Context3 => Graph
                            , Iterator => EdgeIterator
                            , State => WriteableEdgeCostMap
                            , iterEnd => edgeIterEnd
                            , iterNext => edgeIterNext
                            , step => reweightEdgeLoopStep
                            , forLoopRepeat => reweightEdgeLoopRepeat
                            ];

  procedure reweightEdgeLoopStep(obs edgeItr: EdgeIterator,
                                 upd newCosts: WriteableEdgeCostMap,
                                 obs ecm: EdgeCostMap,
                                 obs vcm: VertexCostMap,
                                 obs g: Graph) {
    var currentEdge = edgeIterUnpack(edgeItr);
    call put(newCosts, currentEdge,
             plus(plus(get(ecm, currentEdge),
                       get(vcm, src(currentEdge, g))),
                  negate(get(vcm, tgt(currentEdge, g)))));
  }

  // Dijkstra + adjust utils
  require ForIteratorLoop1_3[ Context1 => WriteableEdgeCostMap
                            , Context2 => VertexCostMap
                            , Context3 => Graph
                            , Iterator => VertexIterator
                            , State => VertexCostMatrix
                            , iterEnd => vertexIterEnd
                            , iterNext => vertexIterNext
                            , step => dijkstraAndAdjustLoopStep
                            , forLoopRepeat => dijkstraAndAdjustLoopRepeat
                            ];

  procedure dijkstraAndAdjustLoopStep(obs vertexItr: VertexIterator,
                                      upd vcmat: VertexCostMatrix,
                                      obs wecm: WriteableEdgeCostMap,
                                      obs vcm: VertexCostMap,
                                      obs g: Graph) {
    var currentVertex = vertexIterUnpack(vertexItr);
    var vcmReweighted = get(vcmat, currentVertex);
    var vpm: VertexPredecessorMap;

    // Call to Dijkstra
    call dijkstraShortestPaths(g, currentVertex, vcmReweighted, wecm, zero(), vpm);

    // Adjust
    var innerVertexItr: VertexIterator;
    call vertices(g, innerVertexItr);
    call adjustVertexLoopRepeat(innerVertexItr, vcmReweighted, currentVertex,
                                vcm);

    // Set the new vertex cost matrix line
    call put(vcmat, currentVertex, vcmReweighted);
  }

  // Adjust utils
  require ForIteratorLoop1_2[ Context1 => VertexDescriptor
                            , Context2 => VertexCostMap
                            , Iterator => VertexIterator
                            , State => VertexCostMap
                            , iterEnd => vertexIterEnd
                            , iterNext => vertexIterNext
                            , step => adjustVertexLoopStep
                            , forLoopRepeat => adjustVertexLoopRepeat
                            ];

  procedure adjustVertexLoopStep(obs vertexItr: VertexIterator,
                                 upd vcmReweighted: VertexCostMap,
                                 obs srcVertex: VertexDescriptor,
                                 obs vcm: VertexCostMap) {
    var currentVertex = vertexIterUnpack(vertexItr);
    call put(vcmReweighted, currentVertex,
             plus(plus(get(vcmReweighted, currentVertex),
                       get(vcm, currentVertex)),
                  negate(get(vcm, srcVertex))));
  }
}

implementation GenericJohnson = {
  use JohnsonBase;
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