package examples.bgl.mg-src.property_map;

concept ReadPropertyMap = {
    require type Key;
    require type Value;

    type PropertyMap;

    function get(pm: PropertyMap, k: Key): Value;

    // TODO: add axiom 'isIn'?
}

concept ReadWritePropertyMap = {
    use ReadPropertyMap;

    procedure put(upd pm: PropertyMap, obs k: Key, obs v: Value);
}
