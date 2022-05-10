import importlib
bgl = importlib.import_module('gen.examples.bgl.mg-src.bgl-py')

PyBellmanFord = bgl.PyBellmanFord()

init_map = PyBellmanFord.initMap
get = PyBellmanFord.get
put = PyBellmanFord.put
make_edge = PyBellmanFord.makeEdge
toVertexDescriptor = PyBellmanFord.toVertexDescriptor

Bool = PyBellmanFord.Bool
Cost = PyBellmanFord.Cost
EdgeCostMap = PyBellmanFord.EdgeCostMap
EdgeDescriptor = PyBellmanFord.EdgeDescriptor
Graph = PyBellmanFord.Graph
Unit = PyBellmanFord.Unit
Vertex = PyBellmanFord.Vertex
VertexCostMap = PyBellmanFord.VertexCostMap
VertexDescriptor = PyBellmanFord.VertexDescriptor
VertexIterator = PyBellmanFord.VertexIterator
VertexPredecessorMap = PyBellmanFord.VertexPredecessorMap

def test_edges_with_cost():
    u, v, x, y, z = 0, 1, 2, 3, 4

    ecm = PyBellmanFord.emptyECMap()
    edges = []

    edges.append(make_edge(Vertex(u), Vertex(y)))
    edges.append(make_edge(Vertex(u), Vertex(x)))
    edges.append(make_edge(Vertex(u), Vertex(v)))
    edges.append(make_edge(Vertex(v), Vertex(u)))
    edges.append(make_edge(Vertex(x), Vertex(y)))
    edges.append(make_edge(Vertex(x), Vertex(v)))
    edges.append(make_edge(Vertex(y), Vertex(v)))
    edges.append(make_edge(Vertex(y), Vertex(z)))
    edges.append(make_edge(Vertex(z), Vertex(u)))
    edges.append(make_edge(Vertex(z), Vertex(x)))

    g = Graph(edges)

    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(u), g), toVertexDescriptor(Vertex(y), g)), Cost(-4.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(u), g), toVertexDescriptor(Vertex(x), g)), Cost(8.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(u), g), toVertexDescriptor(Vertex(v), g)), Cost(5.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(v), g), toVertexDescriptor(Vertex(u), g)), Cost(-2.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(x), g), toVertexDescriptor(Vertex(y), g)), Cost(9.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(x), g), toVertexDescriptor(Vertex(v), g)), Cost(-3.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(y), g), toVertexDescriptor(Vertex(v), g)), Cost(7.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(y), g), toVertexDescriptor(Vertex(z), g)), Cost(2.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(z), g), toVertexDescriptor(Vertex(u), g)), Cost(6.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(z), g), toVertexDescriptor(Vertex(x), g)), Cost(7.0))

    return g, ecm

def bellman_ford_test():
    print('Bellman-Ford test:')
    g, ecm = test_edges_with_cost()

    z = 4
    start, start_cost, other_vertex_base_cost = VertexDescriptor(Vertex(z)), Cost(0), Cost(100.0)

    vitr = VertexIterator()
    PyBellmanFord.vertices(g, vitr)

    vcm = init_map(vitr, other_vertex_base_cost)
    put(vcm, start, start_cost)

    vpm = PyBellmanFord.emptyVPMap()

    allMinimized = Bool()
    u = Unit()

    PyBellmanFord.bellmanFordShortestPaths(g, vcm, ecm, u, vpm, allMinimized)
    
    for k in sorted(vcm.map.keys(), key=lambda k: k.vertex.val):
        print(f'Distance to {k}:', vcm.map[k])
