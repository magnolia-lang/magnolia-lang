import importlib
bgl_v2 = importlib.import_module('gen.examples.bgl_v2.mg-src.bgl_v2-py')

PyBFSTestVisitor = bgl_v2.PyBFSTestVisitor()

cons = PyBFSTestVisitor.cons
emptyVertexList = PyBFSTestVisitor.emptyVertexList
head = PyBFSTestVisitor.head
isEmpty = PyBFSTestVisitor.isEmpty
make_edge = PyBFSTestVisitor.makeEdge
tail = PyBFSTestVisitor.tail

Graph = PyBFSTestVisitor.Graph
EdgeList = PyBFSTestVisitor.EdgeList
Vertex = PyBFSTestVisitor.Vertex
VertexList = PyBFSTestVisitor.VertexList

def test_edges():
    edges = EdgeList()

    edges = cons(make_edge(Vertex(0), Vertex(1)), edges)
    edges = cons(make_edge(Vertex(1), Vertex(2)), edges)
    edges = cons(make_edge(Vertex(1), Vertex(3)), edges)
    edges = cons(make_edge(Vertex(3), Vertex(4)), edges)
    edges = cons(make_edge(Vertex(0), Vertex(4)), edges)
    edges = cons(make_edge(Vertex(4), Vertex(5)), edges)
    edges = cons(make_edge(Vertex(3), Vertex(6)), edges)

    return edges

def bfs_test():
    print('BFS test:')
    edges = test_edges()

    g = Graph(edges)

    start = Vertex(0)

    bfs_result = PyBFSTestVisitor.breadthFirstSearch(g, start,
                                                     emptyVertexList());
    
    ordered_nodes = list()

    while not isEmpty(bfs_result):
        ordered_nodes = [head(bfs_result)] + ordered_nodes
        bfs_result = tail(bfs_result)

    print(ordered_nodes)
