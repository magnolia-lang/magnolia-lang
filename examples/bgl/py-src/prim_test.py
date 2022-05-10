import importlib
bgl = importlib.import_module('gen.examples.bgl.mg-src.bgl-py')

PyPrimVisitor = bgl.PyPrimVisitor()

init_map = PyPrimVisitor.initMap
get = PyPrimVisitor.get
put = PyPrimVisitor.put
make_edge = PyPrimVisitor.makeEdge
toVertexDescriptor = PyPrimVisitor.toVertexDescriptor

Cost = PyPrimVisitor.Cost
EdgeCostMap = PyPrimVisitor.EdgeCostMap
EdgeDescriptor = PyPrimVisitor.EdgeDescriptor
Graph = PyPrimVisitor.Graph
Vertex = PyPrimVisitor.Vertex
VertexCostMap = PyPrimVisitor.VertexCostMap
VertexDescriptor = PyPrimVisitor.VertexDescriptor
VertexIterator = PyPrimVisitor.VertexIterator
VertexPredecessorMap = PyPrimVisitor.VertexPredecessorMap

def test_edges_with_cost():
    ecm = PyPrimVisitor.emptyECMap()
    edges = []
    
    edges.append(make_edge(Vertex(0), Vertex(2)))
    edges.append(make_edge(Vertex(1), Vertex(1)))
    edges.append(make_edge(Vertex(1), Vertex(3)))
    edges.append(make_edge(Vertex(1), Vertex(4)))
    edges.append(make_edge(Vertex(2), Vertex(1)))
    edges.append(make_edge(Vertex(2), Vertex(3)))
    edges.append(make_edge(Vertex(3), Vertex(4)))
    edges.append(make_edge(Vertex(4), Vertex(0)))
    edges.append(make_edge(Vertex(2), Vertex(0)))
    edges.append(make_edge(Vertex(3), Vertex(1)))
    edges.append(make_edge(Vertex(4), Vertex(1)))
    edges.append(make_edge(Vertex(1), Vertex(2)))
    edges.append(make_edge(Vertex(3), Vertex(2)))
    edges.append(make_edge(Vertex(4), Vertex(3)))
    edges.append(make_edge(Vertex(0), Vertex(4)))

    g = Graph(edges)

    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(2), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(1), g)), Cost(2))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(3), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(4), g)), Cost(2))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(2), g), toVertexDescriptor(Vertex(1), g)), Cost(7))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(2), g), toVertexDescriptor(Vertex(3), g)), Cost(3))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(3), g), toVertexDescriptor(Vertex(4), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(4), g), toVertexDescriptor(Vertex(0), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(2), g), toVertexDescriptor(Vertex(0), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(3), g), toVertexDescriptor(Vertex(1), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(4), g), toVertexDescriptor(Vertex(1), g)), Cost(2))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(2), g)), Cost(7))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(3), g), toVertexDescriptor(Vertex(2), g)), Cost(3))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(4), g), toVertexDescriptor(Vertex(3), g)), Cost(1))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(4), g)), Cost(1))

    
    return g, ecm

def prim_test():
    print('Prim test:')
    g, ecm = test_edges_with_cost()

    start, start_cost, other_vertex_base_cost = VertexDescriptor(Vertex(0)), Cost(0), Cost(100.0)

    vitr = VertexIterator()

    PyPrimVisitor.vertices(g, vitr)

    vcm = init_map(vitr, other_vertex_base_cost)
    vpm = PyPrimVisitor.emptyVPMap()

    PyPrimVisitor.primMinimumSpanningTree(g, start, vcm, ecm, start_cost, vpm)
    
    for k in sorted(vpm.map.keys(), key=lambda k: k.vertex.val):
        print(f'Parent of {k}:', vpm.map[k])
