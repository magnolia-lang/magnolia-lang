import importlib
bgl = importlib.import_module('gen.examples.bgl.mg-src.bgl-py')

PyDFSTestVisitor = bgl.PyDFSTestVisitor()

emptyVertexVector = PyDFSTestVisitor.emptyVertexVector
make_edge = PyDFSTestVisitor.makeEdge
to_vertex_descriptor = PyDFSTestVisitor.toVertexDescriptor

Graph = PyDFSTestVisitor.Graph
Vertex = PyDFSTestVisitor.Vertex

def test_edges():
    edges = []

    edges.append(make_edge(Vertex(0), Vertex(1)))
    edges.append(make_edge(Vertex(1), Vertex(2)))
    edges.append(make_edge(Vertex(1), Vertex(3)))
    edges.append(make_edge(Vertex(3), Vertex(4)))
    edges.append(make_edge(Vertex(0), Vertex(4)))
    edges.append(make_edge(Vertex(4), Vertex(5)))
    edges.append(make_edge(Vertex(3), Vertex(6)))

    return edges

def dfs_test():
    print('DFS test:')
    edges = test_edges()

    g = Graph(edges)

    start = to_vertex_descriptor(Vertex(0), g)

    dfs_result = PyDFSTestVisitor.depthFirstSearch(g, start,
                                                     emptyVertexVector());
    
    ordered_nodes = list()

    print(dfs_result.vector)
