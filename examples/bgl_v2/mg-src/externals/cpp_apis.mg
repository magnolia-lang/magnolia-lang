package examples.bgl_v2.mg-src.externals.cpp_apis
    imports examples.bgl_v2.mg-src.externals.signature_apis;

implementation CppBaseTypes = external C++ base.base_types ExtBaseTypes;
implementation CppBaseFloatOps = external C++ base.base_float_ops ExtBaseFloatOps;
implementation CppColorMarker = external C++ base.color_marker ExtColorMarker;
implementation CppList = external C++ base.list ExtList;
implementation CppReadWritePropertyMapWithInitList = external C++ base.read_write_property_map ExtReadWritePropertyMapWithInitList;
implementation CppPair = external C++ base.pair ExtPair;
implementation CppTriplet = external C++ base.triplet ExtTriplet;
implementation CppEdge = external C++ base.edge ExtEdge;
implementation CppIncidenceAndVertexListGraph = external C++ base.incidence_and_vertex_list_graph ExtIncidenceAndVertexListGraph;
implementation CppFIFOQueue = external C++ base.fifo_queue ExtFIFOQueue;
implementation CppUpdateablePriorityQueue =
    external C++ base.priority_queue ExtUpdateablePriorityQueue;
implementation CppWhileLoop = external C++ base.while_loop ExtWhileLoop;
implementation CppWhileLoop4 = external C++ base.while_loop4 ExtWhileLoop4;
