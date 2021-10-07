package examples.bgl_v2.mg-src.dijkstra_utils
    imports examples.bgl_v2.mg-src.bfs
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
    use UpdateablePriorityQueue[ A => Vertex
                               , Priority => Cost
                               , PriorityQueue => PriorityQueue
                               , PriorityMap => VertexCostMap
                               , empty => emptyPriorityQueue
                               , isEmpty => isEmptyQueue
                               ];
    use BFSVisit[ Queue => PriorityQueue
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

    require function makeStateWithMaps(vcm: VertexCostMap,
                                       vpm: VertexPredecessorMap,
                                       ecm: EdgeCostMap): StateWithMaps;
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
        var vpm = getVertexPredecessorMap(swm);
        var ecm = getEdgeCostMap(swm);

        var vcm = origVcm;

        call relax(e, ecm, vcm, vpm);

        if vcm == origVcm
        then skip // nothing changed
        else { // cost diminished or path changed
            swm = putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm));
            pq = update(vcm, tgt(e), pq);
        };
    }

    require VertexListGraph;
    require IncidenceGraph;

    require Pair[ Pair => VertexPair
                , A => Vertex
                , B => Vertex
                , makePair => makeVertexPair
                ];
    require List[ A => VertexPair
                , List => VertexPairList
                , empty => emptyVertexPairList
                ];

    use WhileLoop[ Context => Vertex
                 , State => PopulateVPMapState
                 , cond => populateVPMapLoopCond
                 , step => populateVPMapLoopStep
                 , repeat => populateVPMapLoopRepeat
                 ];

    use Pair[ A => VertexPredecessorMap
            , B => VertexList
            , Pair => PopulateVPMapState
            ];
    
    // TODO: vertex parameter is useless, what to do?
    predicate populateVPMapLoopCond(state: PopulateVPMapState, s: Vertex) =
        !isEmpty(second(state));

    procedure populateVPMapLoopStep(upd state: PopulateVPMapState,
                                    obs s: Vertex) {
        var vpm = first(state);
        var vertexList = second(state);
        var v = head(vertexList);

        state = makePair(put(vpm, v, v), tail(vertexList));
    }

    require function emptyVPMap(): VertexPredecessorMap;

    procedure dijkstraShortestPaths(obs g: Graph,
                                    obs start: Vertex,
                                    upd vcm: VertexCostMap,
                                    obs ecm: EdgeCostMap,
                                    obs initialCost: Cost,
                                    out vpm: VertexPredecessorMap) {
        vcm = put(vcm, start, initialCost);

        var populateVPMapState = makePair(emptyVPMap(), vertices(g));
        call populateVPMapLoopRepeat(populateVPMapState, start);

        vpm = first(populateVPMapState);

        var pq = emptyPriorityQueue(vcm);
        var swm = makeStateWithMaps(vcm, vpm, ecm);
        var c = initMap(vertices(g), white());

        call breadthFirstVisit(g, start, swm, pq, c);

        vcm = getVertexCostMap(swm);
        vpm = getVertexPredecessorMap(swm);

        // TODO
    }
};

implementation GenericDijkstraVisitor = {
    use DijkstraVisitorBase;

    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , defaultAction => discoverVertex
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , defaultAction => examineVertex
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => examineEdge
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => nonTreeEdge
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Edge
                               , defaultAction => blackTarget
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
    use BFSVisitorDefaultAction[ EdgeOrVertex => Vertex
                               , defaultAction => finishVertex
                               , Queue => PriorityQueue
                               , A => StateWithMaps
                               ];
}

// TODO: just cast as program to generate
implementation CppDijkstraVisitorImpl = {
    use GenericDijkstraVisitor;

    use CppColorMarker;
    use CppList[ A => Edge
               , List => EdgeList
               , empty => emptyEdgeList
               ];
    use CppList[ A => Vertex
               , List => VertexList
               , empty => emptyVertexList
               ];
    use CppList[ A => VertexPair
               , List => VertexPairList
               , empty => emptyVertexPairList
               ];
    
    use CppTriplet[ A => StateWithMaps
                  , B => PriorityQueue
                  , C => ColorPropertyMap
                  , Triplet => OuterLoopState
                  , makeTriplet => makeOuterLoopState
                  ];

    use CppTriplet[ A => VertexCostMap
                  , B => VertexPredecessorMap
                  , C => EdgeCostMap
                  , Triplet => StateWithMaps
                  , makeTriplet => makeStateWithMaps
                  , first => getVertexCostMap
                  , second => getVertexPredecessorMap
                  , third => getEdgeCostMap
                  ];

    function putVertexCostMap(vcm: VertexCostMap, swm: StateWithMaps)
        : StateWithMaps = makeStateWithMaps(vcm,
                                            getVertexPredecessorMap(swm),
                                            getEdgeCostMap(swm));

    function putVertexPredecessorMap(vpm: VertexPredecessorMap,
                                     swm: StateWithMaps): StateWithMaps =
        makeStateWithMaps(getVertexCostMap(swm),
                          vpm,
                          getEdgeCostMap(swm));

    use CppPair[ A => OuterLoopState
               , B => EdgeList
               , Pair => InnerLoopState
               , makePair => makeInnerLoopState
               ];

    use CppPair[ A => Graph
               , B => Vertex
               , Pair => InnerLoopContext
               , makePair => makeInnerLoopContext
               ];

    use CppPair[ A => VertexPredecessorMap
               , B => VertexList
               , Pair => PopulateVPMapState
               ];

    use CppPair[ A => Vertex
               , B => Vertex
               , Pair => VertexPair
               , makePair => makeVertexPair
               ];

    use CppUpdateablePriorityQueue[ A => Vertex
                                  , Priority => Cost
                                  , PriorityMap => VertexCostMap
                                  , empty => emptyPriorityQueue
                                  , isEmpty => isEmptyQueue
                                  ];
    //use CppUpdateablePriorityQueue[ A => CostAndVertex ];
    use CppWhileLoop[ Context => Graph
                    , State => OuterLoopState
                    , cond => bfsOuterLoopCond
                    , step => bfsOuterLoopStep
                    , repeat => bfsOuterLoopRepeat
                    ];

    use CppWhileLoop[ Context => InnerLoopContext
                    , State => InnerLoopState
                    , cond => bfsInnerLoopCond
                    , step => bfsInnerLoopStep
                    , repeat => bfsInnerLoopRepeat
                    ];

    use CppWhileLoop[ Context => Vertex
                    , State => PopulateVPMapState
                    , cond => populateVPMapLoopCond
                    , step => populateVPMapLoopStep
                    , repeat => populateVPMapLoopRepeat
                    ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Color
                                           , PropertyMap => ColorPropertyMap
                                           , emptyKeyList => emptyVertexList
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Edge
                                           , KeyList => EdgeList
                                           , Value => Cost
                                           , PropertyMap => EdgeCostMap
                                           , emptyKeyList => emptyEdgeList
                                           , emptyMap => emptyECMap
                                           ];
    
    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Vertex
                                           , PropertyMap => VertexPredecessorMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVPMap
                                           ];

    use CppReadWritePropertyMapWithInitList[ Key => Vertex
                                           , KeyList => VertexList
                                           , Value => Cost
                                           , PropertyMap => VertexCostMap
                                           , emptyKeyList => emptyVertexList
                                           , emptyMap => emptyVCMap
                                           ];

    use CppBaseTypes;
    use CppBaseFloatOps[ Float => Cost ];
    use CppEdge;
    // CppIncidenceAndVertexListGraph exposes the API of both IncidenceGraph
    // and VertexListGraph.
    use CppIncidenceAndVertexListGraph[ consEdgeList => cons
                                      , consVertexList => cons
                                      , headEdgeList => head
                                      , headVertexList => head
                                      , isEmptyEdgeList => isEmpty
                                      , isEmptyVertexList => isEmpty
                                      , tailEdgeList => tail
                                      , tailVertexList => tail
                                      ];
};
