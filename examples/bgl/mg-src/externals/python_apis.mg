package examples.bgl.mg-src.externals.python_apis
    imports examples.bgl.mg-src.externals.signature_apis;

implementation PyBaseTypes = external Python base.base_types ExtBaseTypes;
implementation PyBaseFloatOps = external Python base.base_float_ops ExtBaseFloatOps;
//implementation PyColorMarker = external Python base.color_marker ExtColorMarker;
implementation PyList = external Python base.list_py ExtList;
//implementation PyIterableList = external Python base.iterable_list ExtIterableList;
implementation PyReadWriteColorMapWithInitList = external Python
    base.two_bit_color_map ExtReadWriteColorMapWithInitList;
implementation PyReadWritePropertyMapWithInitList = external Python base.read_write_property_map ExtReadWritePropertyMapWithInitList;
implementation PyPair = external Python base.pair ExtPair;
implementation PyTriplet = external Python base.triplet ExtTriplet;
//implementation PyEdge = external Python base.edge ExtEdge;
implementation PyIncidenceAndVertexListAndEdgeListGraph =
    external Python base.incidence_and_vertex_list_and_edge_list_graph
        ExtIncidenceAndVertexListAndEdgeListGraph;
implementation PyFIFOQueue = external Python base.fifo_queue ExtFIFOQueue;
//implementation PyThreadSafeFIFOQueue =
//    external Python base.thread_safe_fifo_queue ExtFIFOQueue;
implementation PyUpdateablePriorityQueue =
    external Python base.priority_queue ExtUpdateablePriorityQueue;
implementation PyStack = external Python base.stack ExtStack;
implementation PyVector = external Python base.vector ExtVector;
//implementation PyThreadSafeVector =
//    external Python base.thread_safe_vector ExtVector;
implementation PyForIteratorLoop =
    external Python base.for_iterator_loop ExtForIteratorLoop;
implementation PyForIteratorLoop1_2 =
    external Python base.for_iterator_loop1_2 ExtForIteratorLoop1_2;
implementation PyForIteratorLoop1_3 =
    external Python base.for_iterator_loop1_3 ExtForIteratorLoop1_3;
implementation PyForIteratorLoop2_3 =
    external Python base.for_iterator_loop2_3 ExtForIteratorLoop2_3;
implementation PyForIteratorLoop3_2 =
    external Python base.for_iterator_loop3_2 ExtForIteratorLoop3_2;
implementation PyForParallelIteratorLoop3_2 =
    external Python base.for_parallel_iterator_loop3_2 ExtForIteratorLoop3_2;
implementation PyWhileLoop = external Python base.while_loop ExtWhileLoop;
implementation PyWhileLoop3 = external Python base.while_loop3 ExtWhileLoop3;
//implementation PyWhileLoop4_3 = external Python base.while_loop4_3 ExtWhileLoop4_3;

implementation PyBool = external Python base.base_bool ExtBool;
implementation PyUnit = external Python base.base_unit ExtUnit;
