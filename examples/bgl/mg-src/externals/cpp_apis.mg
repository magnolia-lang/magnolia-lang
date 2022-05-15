package examples.bgl.mg-src.externals.cpp_apis
    imports examples.bgl.mg-src.externals.signature_apis;

implementation CppBaseTypes = external C++ base.base_types ExtBaseTypes;
implementation CppBaseFloatOps = external C++ base.base_float_ops ExtBaseFloatOps;
implementation CppColorMarker = external C++ base.color_marker ExtColorMarker;
implementation CppList = external C++ base.list ExtList;
implementation CppIterableList = external C++ base.iterable_list ExtIterableList;
implementation CppReadWriteColorMapWithInitList = external C++
    base.two_bit_color_map ExtReadWriteColorMapWithInitList;
implementation CppReadWritePropertyMapWithInitList = external C++ base.read_write_property_map ExtReadWritePropertyMapWithInitList;
implementation CppPair = external C++ base.pair ExtPair;
implementation CppTriplet = external C++ base.triplet ExtTriplet;
implementation CppEdgeWithoutDescriptor = external C++ base.edge_without_descriptor
    ExtEdgeWithoutDescriptor[ src => srcPlainEdge
                            , tgt => tgtPlainEdge
                            ];
implementation CppCustomIncidenceAndVertexListGraph =
    external C++ base.custom_incidence_and_vertex_list_and_edge_list_graph
        ExtCustomIncidenceAndVertexListAndEdgeListGraph;
implementation CppIncidenceAndVertexListAndEdgeListGraph = external C++ base.incidence_and_vertex_list_and_edge_list_graph ExtIncidenceAndVertexListAndEdgeListGraph;
implementation CppFIFOQueue = external C++ base.fifo_queue ExtFIFOQueue;
implementation CppThreadSafeFIFOQueue =
    external C++ base.thread_safe_fifo_queue ExtFIFOQueue;
implementation CppUpdateablePriorityQueue =
    external C++ base.priority_queue ExtUpdateablePriorityQueue;
implementation CppStack =
    external C++ base.stack ExtStack;
implementation CppVector = external C++ base.vector ExtVector;
implementation CppThreadSafeVector =
    external C++ base.thread_safe_vector ExtVector;
implementation CppForIteratorLoop =
    external C++ base.for_iterator_loop ExtForIteratorLoop;
implementation CppForIteratorLoop1_2 =
    external C++ base.for_iterator_loop1_2 ExtForIteratorLoop1_2;
implementation CppForIteratorLoop1_3 =
    external C++ base.for_iterator_loop1_3 ExtForIteratorLoop1_3;
implementation CppForIteratorLoop2_3 =
    external C++ base.for_iterator_loop2_3 ExtForIteratorLoop2_3;
implementation CppForIteratorLoop3_2 =
    external C++ base.for_iterator_loop3_2 ExtForIteratorLoop3_2;
implementation CppForParallelIteratorLoop3_2 =
    external C++ base.for_parallel_iterator_loop3_2 ExtForIteratorLoop3_2;
implementation CppWhileLoop = external C++ base.while_loop ExtWhileLoop;
implementation CppWhileLoop3 = external C++ base.while_loop3 ExtWhileLoop3;
implementation CppWhileLoop4_3 = external C++ base.while_loop4_3 ExtWhileLoop4_3;

implementation CppBool = external C++ base.base_bool ExtBool;
implementation CppUnit = external C++ base.base_unit ExtUnit;