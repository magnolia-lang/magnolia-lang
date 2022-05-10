package examples.bgl.mg-src.prim_utils
    imports examples.bgl.mg-src.dijkstra_utils;

implementation GenericPrimVisitor = {
  use GenericDijkstraVisitor[ dijkstraShortestPaths => primMinimumSpanningTree
                            , less => less
                            , plus => second
                            ];

  function second(c1: Cost, c2: Cost): Cost = c2;
}