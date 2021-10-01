package examples.bgl.mg-src.v2.property_map;

concept ReadPropertyMap = {
    type PropertyMap;
    type Key;
    type Value;

    function get(pm: PropertyMap, k: Key): Value;

    // TODO: add axiom 'isIn'?
}

concept ReadWritePropertyMap = {
    use ReadPropertyMap;

    function put(pm: PropertyMap, k: Key, v: Value): PropertyMap;
}
