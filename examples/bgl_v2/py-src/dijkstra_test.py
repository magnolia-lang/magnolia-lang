import importlib
bgl_v2 = importlib.import_module('gen.examples.bgl_v2.mg-src.bgl_v2-py')

PyDijkstraVisitor = bgl_v2.PyDijkstraVisitor()

cons = PyDijkstraVisitor.cons
head = PyDijkstraVisitor.head
init_map = PyDijkstraVisitor.initMap
make_edge = PyDijkstraVisitor.makeEdge
get = PyDijkstraVisitor.get
put = PyDijkstraVisitor.put

Cost = PyDijkstraVisitor.Cost
EdgeCostMap = PyDijkstraVisitor.EdgeCostMap
Graph = PyDijkstraVisitor.Graph
EdgeList = PyDijkstraVisitor.EdgeList
Vertex = PyDijkstraVisitor.Vertex
VertexCostMap = PyDijkstraVisitor.VertexCostMap
VertexPredecessorMap = PyDijkstraVisitor.VertexPredecessorMap

def test_edges_with_cost():
    ecm = PyDijkstraVisitor.emptyECMap()
    edges = EdgeList()
    
    edges = cons(make_edge(Vertex(0), Vertex(1)), edges)
    ecm = put(ecm, head(edges), Cost(0.5))

    edges = cons(make_edge(Vertex(1), Vertex(2)), edges)
    ecm = put(ecm, head(edges), Cost(5.6))

    edges = cons(make_edge(Vertex(1), Vertex(3)), edges)
    ecm = put(ecm, head(edges), Cost(0.2))

    edges = cons(make_edge(Vertex(3), Vertex(4)), edges)
    ecm = put(ecm, head(edges), Cost(0.1))

    edges = cons(make_edge(Vertex(0), Vertex(4)), edges)
    ecm = put(ecm, head(edges), Cost(3.2))

    return edges, ecm

def dijkstra_test():
    print('Dijkstra test:')
    edges, ecm = test_edges_with_cost()

    g = Graph(edges)

    start, start_cost, other_vertex_base_cost = Vertex(0), Cost(0), Cost(100.0)

    vcm = init_map(PyDijkstraVisitor.vertices(g), other_vertex_base_cost)
    vpm = PyDijkstraVisitor.emptyVPMap()

    PyDijkstraVisitor.dijkstraShortestPaths(g, start, vcm, ecm, start_cost,
                                            vpm)
    
    for k in sorted(vcm.map.keys(), key=lambda k: k.val):
        print(f'Distance to {k}:', vcm.map[k])
