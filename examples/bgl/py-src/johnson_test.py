import importlib
bgl = importlib.import_module('gen.examples.bgl.mg-src.bgl-py')

PyJohnson = bgl.PyJohnson()

init_map = PyJohnson.initMap
get = PyJohnson.get
put = PyJohnson.put
emptyVCMap = PyJohnson.emptyVCMap
emptyVCMatrix = PyJohnson.emptyVCMatrix
make_edge = PyJohnson.makeEdge
toVertexDescriptor = PyJohnson.toVertexDescriptor

Bool = PyJohnson.Bool
Cost = PyJohnson.Cost
EdgeCostMap = PyJohnson.EdgeCostMap
EdgeDescriptor = PyJohnson.EdgeDescriptor
Graph = PyJohnson.Graph
Unit = PyJohnson.Unit
Vertex = PyJohnson.Vertex
VertexCostMap = PyJohnson.VertexCostMap
VertexDescriptor = PyJohnson.VertexDescriptor
VertexIterator = PyJohnson.VertexIterator
VertexPredecessorMap = PyJohnson.VertexPredecessorMap

def test_edges_with_cost():
    ecm = PyJohnson.emptyECMap()
    edges = []

    edges.append(make_edge(Vertex(0), Vertex(1)))
    edges.append(make_edge(Vertex(0), Vertex(4)))
    edges.append(make_edge(Vertex(0), Vertex(2)))
    edges.append(make_edge(Vertex(1), Vertex(3)))
    edges.append(make_edge(Vertex(1), Vertex(4)))
    edges.append(make_edge(Vertex(2), Vertex(1)))
    edges.append(make_edge(Vertex(3), Vertex(2)))
    edges.append(make_edge(Vertex(3), Vertex(0)))
    edges.append(make_edge(Vertex(4), Vertex(2)))

    g = Graph(edges)

    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(1), g)), Cost(3.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(4), g)), Cost(-4.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(0), g), toVertexDescriptor(Vertex(2), g)), Cost(8.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(3), g)), Cost(1.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(1), g), toVertexDescriptor(Vertex(4), g)), Cost(7.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(2), g), toVertexDescriptor(Vertex(1), g)), Cost(4.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(3), g), toVertexDescriptor(Vertex(2), g)), Cost(-5.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(3), g), toVertexDescriptor(Vertex(0), g)), Cost(2.0))
    put(ecm, EdgeDescriptor(toVertexDescriptor(Vertex(4), g), toVertexDescriptor(Vertex(2), g)), Cost(6.0))

    return g, ecm

def johnson_test():
    print('Johnson test:')
    g, ecm = test_edges_with_cost()

    vitr = VertexIterator()
    PyJohnson.vertices(g, vitr)

    vcmat = emptyVCMatrix()
    
    while not PyJohnson.vertexIterEnd(vitr):
        put(vcmat, PyJohnson.vertexIterUnpack(vitr), emptyVCMap())
        PyJohnson.vertexIterNext(vitr)

    success = Bool()
    u = Unit()

    PyJohnson.johnsonAllPairsShortestPaths(g, ecm, u, vcmat, success)

    v = lambda i: toVertexDescriptor(Vertex(i), g)

    for i in range(0, 5):
        for j in range(0, 5):
            print('Shortest distance from', i, f'to {j}:', get(get(vcmat, v(i)), v(j)))
