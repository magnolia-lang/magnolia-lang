import importlib
bgl_v2 = importlib.import_module('gen.examples.bgl_v2.mg-src.bgl_v2-py')

PyBFSTestVisitor = bgl_v2.PyBFSTestVisitor()

emptyVertexVector = PyBFSTestVisitor.emptyVertexVector
make_edge = PyBFSTestVisitor.makeEdge
to_vertex_descriptor = PyBFSTestVisitor.toVertexDescriptor

Graph = PyBFSTestVisitor.Graph
Vertex = PyBFSTestVisitor.Vertex

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

def bfs_test():
    print('BFS test:')
    edges = test_edges()

    g = Graph(edges)

    start = to_vertex_descriptor(Vertex(0), g)

    bfs_result = PyBFSTestVisitor.breadthFirstSearch(g, start,
                                                     emptyVertexVector());
    
    ordered_nodes = list()

    print(bfs_result.vector)
