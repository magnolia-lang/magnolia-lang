package examples.bgl.mg-src.bgl
    imports examples.bgl.mg-src.BasicConcepts
          , examples.bgl.mg-src.ExternalDataStructures
          , examples.bgl.mg-src.GraphConcepts;

concept IterableCollection = {
    type Collection;
    type E;

    function emptyCollection(): Collection;
    function addToCollection(c: Collection, e: E): Collection;

    function extractOneElement(c: Collection): E
        guard !isCollectionEmpty(c);
    function removeFromCollection(c: Collection, e: E): Collection
        guard isIn(c, e);

    predicate isCollectionEmpty(c: Collection);
    predicate isIn(c: Collection, e: E);
}

implementation GraphBFS = {
    require type Edge;
    require type EdgeCollection;
    require type Graph;
    require type Vertex;
    require type VertexCollection;
    require IncidenceGraph;

    require Queue[ Queue => VertexQueue, E => Vertex ];
    require IterableCollection[ Collection => VertexCollection
                              , E => Vertex
                              , emptyCollection => emptyVertexCollection
                              ];

    // outer loop requirements
    require Twople[ A => VertexCollection
                  , B => VertexQueue
                  , Twople => OuterStateTwople
                  ];
    require WhileLoop[ condition => mainLoopCondition
                     , repeat => mainLoop
                     , State => OuterStateTwople
                     , Context => Graph
                     , step => visitVertex
                     ];

    // inner loop requirements
    require Triple[ A => VertexCollection
                  , B => VertexQueue
                  , C => VertexCollection
                  , Triple => InnerStateTriple
                  ];
    require WhileLoop[ condition => addToQueueLoopCondition
                     , repeat => addToQueueLoop
                     , State => InnerStateTriple
                     , Context => Graph
                     , step => addToQueueStep
                     ];

    predicate mainLoopCondition(st: OuterStateTwople, _: Graph) {
        var vertexQueue = snd(st);
        value !isQueueEmpty(vertexQueue);
    }

    predicate addToQueueLoopCondition(st: InnerStateTriple, g: Graph) {
        value !isCollectionEmpty(fst(st));
    }

    function emptyVertexCollection(): VertexCollection;
    function addToCollection(vc: VertexCollection, v: Vertex): VertexCollection;

    // adjacentVertices requirements
    require CollectionMapFunction[ A => Edge
                                 , B => Vertex
                                 , CollectionA => EdgeCollection
                                 , CollectionB => VertexCollection
                                 , map => mapEdgeTarget
                                 , f => target
                                 ];

    function adjacentVertices(g: Graph, v: Vertex): VertexCollection =
        mapEdgeTarget(outEdges(g, v));

    procedure addToQueueStep(upd st: InnerStateTriple, obs g: Graph) {
        var vertexQueue = snd(st);
        var visitedVertices = third(st);
        var nextVertex = extractOneElement(fst(st));

        var newVertexQueue = if isIn(visitedVertices, nextVertex)
                             then snd(st)
                             else enqueue(snd(st), nextVertex);
        var newVisitedVertices = addToCollection(visitedVertices, nextVertex);
        // create next iteration state
        st = triple(removeFromCollection(fst(st), nextVertex), newVertexQueue, newVisitedVertices);
    }

    procedure visitVertex(upd st: OuterStateTwople, obs g: Graph) {
        var v = first(snd(st));
        // perform some visit action if not yet visited
        call visitAction(v);
        // add to collection of visited vertices
        var visitedVertices = addToCollection(fst(st), v);
        // add new vertices to the queue
        //  a) create state
        var innerState = triple(adjacentVertices(g, v), snd(st), visitedVertices);
        //  b) inner loop
        call addToQueueLoop(innerState, g);
        var newVisitedVertices = third(innerState);
        // create next iteration queue and state
        var newQueue = dequeue(snd(innerState));
        st = twople(newVisitedVertices, newQueue);
    }

    // not passing around the world state here, but perhaps we should
    procedure visitAction(obs v: Vertex);

    procedure bfs(obs g: Graph, obs start: Vertex) {
        var queue = enqueue(emptyQueue(), start);
        var visitedVertices = emptyVertexCollection();

        var outerState = twople(visitedVertices, queue);
        call mainLoop(outerState, g);
    }
}

program IncidenceGraphWalk = {
    use GraphBFS[ Edge => Edge
                , EdgeCollection => EdgeCollection
                , InnerStateTriple => InnerStateTriple
                , OuterStateTwople => OuterStateTwople
                , Vertex => StringVertex
                , VertexCollection => VertexCollection
                , VertexQueue => VertexQueue
                , fst => first
                , snd => second
                , third => third
                , visitAction => pprint
                ];
    use CxxQueue[ Queue => VertexQueue
                , E => StringVertex
                , empty => isQueueEmpty
                , nil => emptyQueue
                ];
    use CxxEdge[ Edge => Edge // lacking extend
               , Vertex => StringVertex
               , source => source
               , target => target
               ];
    use CxxHashSet[ HashSet => EdgeCollection // lacking extend
                  , E => Edge
                  , nil => emptyEdgeCollection
                  , insert => addToCollection
                  , remove => removeFromCollection
                  , min => extractOneElement
                  , member => isIn
                  , empty => isCollectionEmpty
                  ];
    use CxxIncidenceGraph[ Edge => Edge
                         , EdgeCollection => EdgeCollection
                         , Graph => Graph
                         , Vertex => StringVertex
                         , outEdges => outEdges
                         , source => source
                         , target => target
                         ];
    use CxxMapFunction[ B => StringVertex
                      , CollectionB => VertexCollection
                      , A => Edge
                      , CollectionA => EdgeCollection
                      , map => mapEdgeTarget
                      , f => target
                      , emptyCollectionB => emptyVertexCollection
                      ];
    use CxxString[ String => StringVertex ];
    use CxxTuple3[ Tuple => InnerStateTriple
                 , first => first
                 , second => second
                 , third => third
                 , make_tuple => triple
                 , A => VertexCollection
                 , B => VertexQueue
                 , C => VertexCollection
                 ];
    use CxxPair[ Pair => OuterStateTwople
               , first => first
               , second => second
               , make_pair => twople
               , A => VertexCollection
               , B => VertexQueue
               ];
    use CxxHashSet[ HashSet => VertexCollection
                  , E => StringVertex
                  , nil => emptyVertexCollection
                  , insert => addToCollection
                  , remove => removeFromCollection
                  , min => extractOneElement
                  , member => isIn
                  , empty => isCollectionEmpty
                  ];
    use CxxPPrinter[ E => StringVertex ];
    // TODO: handle required functions generation, implement
    use CxxWhileLoop[ cond => mainLoopCondition
                    , State => OuterStateTwople
                    , Context => Graph
                    , repeat => mainLoop
                    , body => visitVertex
                    ];
    use CxxWhileLoop[ cond => addToQueueLoopCondition
                    , State => InnerStateTriple
                    , Context => Graph
                    , repeat => addToQueueLoop
                    , body => addToQueueStep
                    ];
}

/*
program AdjacencyGraphWalk = {
    use GraphBFS[ Vertex => StringVertex
                , fst => first
                , snd => second
                , third => third
                , visitAction => pprint
                ];
    use CxxQueue[ Queue => VertexQueue
                , E => StringVertex
                , empty => isQueueEmpty
                , nil => emptyQueue
                ];
    use CxxAdjacencyGraph[Vertex => StringVertex];
    use CxxString[String => StringVertex];
    use CxxTuple3[ Tuple => InnerStateTriple
                 , first => first
                 , second => second
                 , third => third
                 , make_tuple => triple
                 , A => VertexCollection
                 , B => VertexQueue
                 , C => VertexCollection
                 ];
    use CxxPair[ Pair => OuterStateTwople
               , first => first
               , second => second
               , make_pair => twople
               , A => VertexCollection
               , B => VertexQueue
               ];
    use CxxHashSet[ HashSet => VertexCollection
                  , E => StringVertex
                  , nil => emptyCollection
                  , insert => addToCollection
                  , remove => removeFromCollection
                  , min => extractOneElement
                  , member => isIn
                  , empty => isCollectionEmpty
                  ];
    use CxxPPrinter[ E => StringVertex ];
    // TODO: handle required functions generation, implement
    use CxxWhileLoop[ cond => mainLoopCondition
                    , State => OuterStateTwople
                    , Context => Graph
                    , repeat => mainLoop
                    , body => visitVertex
                    ];
    use CxxWhileLoop[ cond => addToQueueLoopCondition
                    , State => InnerStateTriple
                    , Context => Graph
                    , repeat => addToQueueLoop
                    , body => addToQueueStep
                    ];
    use CxxUnit[ Unit => Unit ];
}*/
