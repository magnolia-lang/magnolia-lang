import importlib
bgl = importlib.import_module('gen.examples.bgl.mg-src.bgl-py')

PyDijkstraVisitor = bgl.PyDijkstraVisitor()

init_map = PyDijkstraVisitor.initMap
get = PyDijkstraVisitor.get
put = PyDijkstraVisitor.put
make_edge = PyDijkstraVisitor.makeEdge
toVertexDescriptor = PyDijkstraVisitor.toVertexDescriptor

Cost = PyDijkstraVisitor.Cost
EdgeCostMap = PyDijkstraVisitor.EdgeCostMap
EdgeDescriptor = PyDijkstraVisitor.EdgeDescriptor
Graph = PyDijkstraVisitor.Graph
Vertex = PyDijkstraVisitor.Vertex
VertexCostMap = PyDijkstraVisitor.VertexCostMap
VertexDescriptor = PyDijkstraVisitor.VertexDescriptor
VertexIterator = PyDijkstraVisitor.VertexIterator
VertexPredecessorMap = PyDijkstraVisitor.VertexPredecessorMap

def test_edges_with_cost():
    ecm = PyDijkstraVisitor.emptyECMap()
    edges = []
    
    edges.append(make_edge(Vertex(0), Vertex(1)))
    edges.append(make_edge(Vertex(1), Vertex(2)))
    edges.append(make_edge(Vertex(1), Vertex(3)))
    edges.append(make_edge(Vertex(3), Vertex(4)))
    edges.append(make_edge(Vertex(0), Vertex(4)))

    g = Graph(edges)

    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(1), g)), Cost(0.5))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(2), g)), Cost(5.6))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(3), g)), Cost(0.2))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(3), g), toVertexDescriptor(Vertex(4), g)), Cost(0.1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(4), g)), Cost(3.2))
    
    return g, ecm

def dijkstra_test():
    print('Dijkstra test:')
    g, ecm = test_edges_with_cost()

    start, start_cost, other_vertex_base_cost = VertexDescriptor(Vertex(0)), Cost(0), Cost(100.0)

    vitr = VertexIterator()

    PyDijkstraVisitor.vertices(g, vitr)

    vcm = init_map(vitr, other_vertex_base_cost)
    vpm = PyDijkstraVisitor.emptyVPMap()

    PyDijkstraVisitor.dijkstraShortestPaths(g, start, vcm, ecm, start_cost,
                                            vpm)
    
    for k in sorted(vcm.map.keys(), key=lambda k: k.vertex.val):
        print(f'Distance to {k}:', vcm.map[k])
