package examples.bgl_v2.mg-src.property_map;

concept ReadPropertyMap = {
    require type Key;
    require type Value;

    type PropertyMap;

    function get(pm: PropertyMap, k: Key): Value;

    // TODO: add axiom 'isIn'?
}

concept ReadWritePropertyMap = {
    use ReadPropertyMap;

    function put(pm: PropertyMap, k: Key, v: Value): PropertyMap;
}
