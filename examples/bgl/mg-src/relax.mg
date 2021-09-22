package examples.bgl.mg-src.relax
    imports examples.bgl.mg-src.graph
          , examples.bgl.mg-src.property_map;

implementation CombinableAndComparable = {
    type A;
    function combine(a1: A, a2: A): A;
    predicate compare(a1: A, a2: A);
}

implementation Relax = {
    
    use CombinableAndComparable[ A => Cost
                               , combine => plus
                               , compare => less
                               ];
    use Edge[ Edge => Edge
            , EdgeDescriptor => EdgeDescriptor
            , Vertex => Vertex
            , VertexDescriptor => VertexDescriptor
            ];

    use ReadWritePropertyMap[ Key => VertexDescriptor
                            , Value => Cost
                            , PropertyMap => VertexCostMap
                            ];
    use ReadWritePropertyMap[ Key => VertexDescriptor
                            , Value => VertexDescriptor
                            , PropertyMap => VertexPredecessorMap
                            ];
    use ReadPropertyMap[ Key => EdgeDescriptor
                       , Value => Cost
                       , PropertyMap => EdgeCostMap
                       ];

    procedure relax(obs e: EdgeDescriptor, obs g: Graph, obs ecm: EdgeCostMap,
                    upd vcm: VertexCostMap, upd vpm: VertexPredecessorMap) {
        var u = src(e, g);
        var v = tgt(e, g);

        var uCost = get(vcm, u);
        var vCost = get(vcm, v);

        var edgeCost = get(ecm, e);

        if less(plus(uCost, edgeCost), vCost)
        then {
            call put(vcm, v, plus(uCost, edgeCost));
            call put(vpm, v, u);
        }
        else skip;
    }
}
