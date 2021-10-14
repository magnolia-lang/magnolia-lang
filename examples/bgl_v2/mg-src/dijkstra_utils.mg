package examples.bgl_v2.mg-src.dijkstra_utils
    imports examples.bgl_v2.mg-src.bfs
          , examples.bgl_v2.mg-src.for_loop
          // TODO: the following line triggers a compiler bug if removed and
          //       we write 'require VertexListGraph', for instance.
          , examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.list
          , examples.bgl_v2.mg-src.queue
          , examples.bgl_v2.mg-src.relax
          , examples.bgl_v2.mg-src.tuple
          , examples.bgl_v2.mg-src.while_loop
          , examples.bgl_v2.mg-src.externals.cpp_apis;

implementation DijkstraVisitorBase = {
    use UpdateablePriorityQueue[ A => VertexDescriptor
                               , Priority => Cost
                               , PriorityQueue => PriorityQueue
                               , PriorityMap => VertexCostMap
                               , empty => emptyPriorityQueue
                               , isEmpty => isEmptyQueue
                               ];
    use BFSVisit[ Queue => PriorityQueue
                , A => StateWithMaps
                ]; // TODO: the rest

    use Relax[ Edge => Edge
             , EdgeDescriptor => EdgeDescriptor
             , Vertex => Vertex
             , VertexDescriptor => VertexDescriptor
             ];

    require function getVertexCostMap(s: StateWithMaps): VertexCostMap;
    require function putVertexCostMap(vcm: VertexCostMap,
                                      s: StateWithMaps): StateWithMaps;

    require function getVertexPredecessorMap(s: StateWithMaps)
        : VertexPredecessorMap;
    require function putVertexPredecessorMap(vcm: VertexPredecessorMap,
                                             s: StateWithMaps): StateWithMaps;

    require function getEdgeCostMap(s: StateWithMaps): EdgeCostMap;

    require function makeStateWithMaps(vcm: VertexCostMap,
                                       vpm: VertexPredecessorMap,
                                       ecm: EdgeCostMap): StateWithMaps;
    procedure treeEdge(obs e: EdgeDescriptor,
                       obs g: Graph,
                       upd pq: PriorityQueue,
                       upd swm: StateWithMaps) {
        
        var vcm = getVertexCostMap(swm);
        var vpm = getVertexPredecessorMap(swm);
        var ecm = getEdgeCostMap(swm);

        call relax(e, g, ecm, vcm, vpm);

        swm = putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm));
    }

    procedure grayTarget(obs e: EdgeDescriptor,
                         obs g: Graph,
                         upd pq: PriorityQueue,
                         upd swm: StateWithMaps) {
        var origVcm = getVertexCostMap(swm);
        var vpm = getVertexPredecessorMap(swm);
        var ecm = getEdgeCostMap(swm);

        var vcm = origVcm;

        call relax(e, g, ecm, vcm, vpm);

        if vcm == origVcm
        then skip // nothing changed
        else { // cost diminished or path changed
            swm = putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm));
            pq = update(vcm, tgt(e, g), pq);
        };
    }

    require VertexListGraph;
    require IncidenceGraph;

    use ForIteratorLoop[ Context => VertexDescriptor
                       , Iterator => VertexIterator
                       , State => VertexPredecessorMap
                       , iterNext => vertexIterNext
                       , step => populateVPMapLoopStep
                       , forLoopRepeat => populateVPMapLoopRepeat
                       ];

    procedure populateVPMapLoopStep(obs itr: VertexIterator,
                                    obs endItr: VertexIterator,
                                    upd vpm: VertexPredecessorMap,
                                    obs vd: VertexDescriptor) {
        var v = vertexIterUnpack(itr);
        call put(vpm, v, v);
    }

    require function emptyVPMap(): VertexPredecessorMap;

    procedure dijkstraShortestPaths(obs g: Graph,
                                    obs start: VertexDescriptor,
                                    upd vcm: VertexCostMap,
                                    obs ecm: EdgeCostMap,
                                    obs initialCost: Cost,
                                    out vpm: VertexPredecessorMap) {
        call put(vcm, start, initialCost);

        var vertexBeg: VertexIterator;
        var vertexEnd: VertexIterator;
        call vertices(g, vertexBeg, vertexEnd);

        vpm = emptyVPMap();

        call populateVPMapLoopRepeat(vertexBeg, vertexEnd, vpm, start);

        var pq = emptyPriorityQueue(vcm);
        var swm = makeStateWithMaps(vcm, vpm, ecm);
        var c = initMap(vertexBeg, vertexEnd, white());

        call breadthFirstVisit(g, start, swm, pq, c);

        vcm = getVertexCostMap(swm);
        vpm = getVertexPredecessorMap(swm);

        // TODO
    }

    function putVertexCostMap(vcm: VertexCostMap, swm: StateWithMaps)
        : StateWithMaps = makeStateWithMaps(vcm,
                                            getVertexPredecessorMap(swm),
                                            getEdgeCostMap(swm));

    function putVertexPredecessorMap(vpm: VertexPredecessorMap,
                                     swm: StateWithMaps): StateWithMaps =
        makeStateWithMaps(getVertexCostMap(swm),
                          vpm,
                          getEdgeCostMap(swm));
};

implementation GenericDijkstraVisitor = {
    use DijkstraVisitorBase;

    use BFSVisitorDefaultAction[ EdgeOrVertex => VertexDescriptor
                               , defaultAction => discoverVertex
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => VertexDescriptor
                               , defaultAction => examineVertex
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => EdgeDescriptor
                               , defaultAction => examineEdge
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => EdgeDescriptor
                               , defaultAction => nonTreeEdge
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => EdgeDescriptor
                               , defaultAction => blackTarget
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => VertexDescriptor
                               , defaultAction => finishVertex
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
}
