package examples.bgl_v2.mg-src.externals.python_apis
    imports examples.bgl_v2.mg-src.externals.signature_apis;

implementation PyBaseTypes = external Python base.base_types ExtBaseTypes;
implementation PyBaseFloatOps = external Python base.base_float_ops ExtBaseFloatOps;
implementation PyColorMarker = external Python base.color_marker ExtColorMarker;
implementation PyList = external Python base.list_py ExtList;
implementation PyReadWritePropertyMapWithInitList = external Python base.read_write_property_map ExtReadWritePropertyMapWithInitList;
implementation PyPair = external Python base.pair ExtPair;
implementation PyTriplet = external Python base.triplet ExtTriplet;
implementation PyEdge = external Python base.edge ExtEdge;
implementation PyIncidenceAndVertexListGraph = external Python base.incidence_and_vertex_list_graph ExtIncidenceAndVertexListGraph;
implementation PyFIFOQueue = external Python base.fifo_queue ExtFIFOQueue;
implementation PyUpdateablePriorityQueue =
    external Python base.priority_queue ExtUpdateablePriorityQueue;
implementation PyWhileLoop = external Python base.while_loop ExtWhileLoop;
