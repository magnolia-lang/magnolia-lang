package examples.bgl.mg-src.v2.externals.cpp.property_map_cpp
    imports examples.bgl.mg-src.v2.property_map;

implementation CppReadWritePropertyMapWithInitList =
    external C++ base.read_write_property_map {
    
    use signature(ReadWritePropertyMap);

    require type KeyList;
    function initMap(kl: KeyList, v: Value): PropertyMap;

}

