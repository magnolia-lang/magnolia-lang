package examples.bgl_v2.mg-src.dijkstra
    imports examples.bgl_v2.mg-src.bfs
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.relax;

implementation DijkstraVisitor = {
    use UpdateablePriorityQueue[ A => Vertex
                               , Priority => Cost
                               , PriorityQueue => PriorityQueue
                               , PriorityMap => VertexCostMap
                               ];
    use BFSVisitor[ Queue => PriorityQueue
                  , A => StateWithMaps
                  ]; // TODO: the rest
    use Relax;

    require function getVertexCostMap(s: StateWithMaps): VertexCostMap;
    require function putVertexCostMap(vcm: VertexCostMap,
                                      s: StateWithMaps): StateWithMaps;

    require function getVertexPredecessorMap(s: StateWithMaps)
        : VertexPredecessorMap;
    require function putVertexPredecessorMap(vcm: VertexPredecessorMap,
                                             s: StateWithMaps): StateWithMaps;

    require function getEdgeCostMap(s: StateWithMaps): EdgeCostMap;

    procedure treeEdge(obs e: Edge,
                       obs g: Graph,
                       upd pq: PriorityQueue,
                       upd swm: StateWithMaps) {
        
        var vcm = getVertexCostMap(swm);
        var vpm = getVertexPredecessorMap(swm);
        var ecm = getEdgeCostMap(swm);

        call relax(e, ecm, vcm, vpm);

        swm = putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm));
    }

    procedure grayTarget(obs e: Edge,
                         obs g: Graph,
                         upd pq: PriorityQueue,
                         upd swm: StateWithMaps) {
        var origVcm = getVertexCostMap(swm);
        var origVpm = getVertexPredecessorMap(swm);
        var ecm = getEdgeCostMap(swm);

        var vcm = origVcm;
        var vpm = origVpm;

        call relax(e, ecm, vcm, vpm);

        if vcm == origVcm
        then skip // nothing changed
        else { // cost diminished or path changed
            swm = putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm));
            pq = update(vcm, tgt(e), pq);
        };
    }
};
