package examples.bgl_v2.mg-src.externals.cpp.property_map_cpp
    imports examples.bgl_v2.mg-src.property_map;

implementation CppReadWritePropertyMapWithInitList =
    external C++ base.read_write_property_map {
    
    use signature(ReadWritePropertyMap);

    require type KeyList;
    function initMap(kl: KeyList, v: Value): PropertyMap;

}

