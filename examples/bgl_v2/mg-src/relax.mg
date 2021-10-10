package examples.bgl_v2.mg-src.relax
    imports examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.property_map;

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
            , Vertex => Vertex
            ];
    use ReadWritePropertyMap[ Key => Vertex
                            , Value => Cost
                            , PropertyMap => VertexCostMap
                            ];
    use ReadWritePropertyMap[ Key => Vertex
                            , Value => Vertex
                            , PropertyMap => VertexPredecessorMap
                            ];
    use ReadPropertyMap[ Key => Edge
                       , Value => Cost
                       , PropertyMap => EdgeCostMap
                       ];
    use Edge;

    procedure relax(obs e: Edge, obs ecm: EdgeCostMap, upd vcm: VertexCostMap,
                    upd vpm: VertexPredecessorMap) {
        var u = src(e);
        var v = tgt(e);

        var uCost = get(vcm, u);
        var vCost = get(vcm, v);

        var edgeCost = get(ecm, e);

        if less(plus(uCost, edgeCost), vCost)
        then {
            put(vcm, v, plus(uCost, edgeCost));
            put(vpm, v, u);
        }
        else skip;
    }
}
