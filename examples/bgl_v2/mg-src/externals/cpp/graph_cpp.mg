package examples.bgl_v2.mg-src.externals.cpp.graph_cpp
    imports examples.bgl_v2.mg-src.graph
          , examples.bgl_v2.mg-src.externals.cpp.list_cpp;

// TODO: there is a lot of requirements to pass through because we do not have
// the extend keyword. Perhaps this should be improved.

implementation CppEdge = external C++ base.edge signature(Edge);

implementation CppGraphType = external C++ base.graph {
    require CppEdge;

    type Graph;
}

implementation CppIncidenceGraph = external C++ base.incidence_graph {
    require CppGraphType;
    require type VertexCount;

    require CppList[ A => Edge
                   , List => EdgeList
                   ];

    function outEdges(v: Vertex, g: Graph): EdgeList;
    function outDegree(v: Vertex, g: Graph): VertexCount;
}

implementation CppVertexListGraph = external C++ base.vertex_list_graph {
    require CppGraphType;
    require CppList[ A => Vertex
                   , List => VertexList
                   ];
    require type VertexCount;

    function vertices(g: Graph): VertexList;
    function numVertices(g: Graph): VertexCount;
}
