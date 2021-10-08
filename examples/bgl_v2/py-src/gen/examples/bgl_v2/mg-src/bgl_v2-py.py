from base import base_float_ops
from base import base_types
from base import color_marker
from base import edge
from base import fifo_queue
from base import incidence_and_vertex_list_graph
from base import list_py
from base import pair
from base import priority_queue
from base import read_write_property_map
from base import triplet
from base import while_loop
from collections import namedtuple
import functools
import multiple_dispatch


def PyBFSTestVisitor():
    overload = functools.partial(multiple_dispatch.overload, {})
    __color_marker = color_marker()
    __base_types = base_types()
    Vertex = __base_types.Vertex
    __edge = edge(Vertex)
    __fifo_queue = fifo_queue(Vertex)
    __list_py0 = list_py(Vertex)
    VertexList = __list_py0.List
    @overload(return_type=VertexList)
    def emptyVertexList():
        return __list_py0.empty()

    @overload(VertexList, return_type=bool)
    def isEmpty(l):
        return __list_py0.isEmpty(l)

    @overload(VertexList, return_type=VertexList)
    def tail(l):
        return __list_py0.tail(l)

    @overload(Vertex, VertexList, return_type=VertexList)
    def cons(a, l):
        return __list_py0.cons(a, l)

    @overload(VertexList, return_type=Vertex)
    def head(l):
        return __list_py0.head(l)

    Int = __base_types.Int
    FIFOQueue = __fifo_queue.FIFOQueue
    @overload(return_type=FIFOQueue)
    def empty():
        return __fifo_queue.empty()

    @overload(FIFOQueue, return_type=Vertex)
    def front(q):
        return __fifo_queue.front(q)

    @overload(FIFOQueue, return_type=bool)
    def isEmptyQueue(q):
        return __fifo_queue.isEmpty(q)

    @overload(FIFOQueue, return_type=FIFOQueue)
    def pop(q):
        return __fifo_queue.pop(q)

    @overload(Vertex, FIFOQueue, return_type=FIFOQueue)
    def push(a, q):
        return __fifo_queue.push(a, q)

    Edge = __edge.Edge
    __list_py = list_py(Edge)
    EdgeList = __list_py.List
    @overload(return_type=EdgeList)
    def emptyEdgeList():
        return __list_py.empty()

    @overload(EdgeList, return_type=bool)
    def isEmpty(l):
        return __list_py.isEmpty(l)

    @overload(EdgeList, return_type=EdgeList)
    def tail(l):
        return __list_py.tail(l)

    @overload(Edge, EdgeList, return_type=EdgeList)
    def cons(a, l):
        return __list_py.cons(a, l)

    @overload(EdgeList, return_type=Edge)
    def head(l):
        return __list_py.head(l)

    @overload(Vertex, Vertex, return_type=Edge)
    def makeEdge(s, t):
        return __edge.makeEdge(s, t)

    @overload(Edge, return_type=Vertex)
    def src(e):
        return __edge.src(e)

    @overload(Edge, return_type=Vertex)
    def tgt(e):
        return __edge.tgt(e)

    __incidence_and_vertex_list_graph = incidence_and_vertex_list_graph(Edge, EdgeList, Vertex, VertexList, cons, cons, emptyEdgeList, emptyVertexList, head, head, isEmpty, isEmpty, makeEdge, src, tail, tail, tgt)
    Graph = __incidence_and_vertex_list_graph.Graph
    __pair = pair(Graph, Vertex)
    InnerLoopContext = __pair.Pair
    @overload(InnerLoopContext, return_type=Vertex)
    def second(p):
        return __pair.second(p)

    @overload(Graph, Vertex, VertexList, return_type=VertexList)
    def breadthFirstSearch(g, start, init):
        q = empty(return_type=FIFOQueue)
        c = initMap(vertices(g, return_type=VertexList), white(return_type=Color), return_type=ColorPropertyMap)
        a = init
        breadthFirstVisit(g, start, a, q, c, return_type=None)
        return a

    @overload(Vertex, Graph, FIFOQueue, VertexList, return_type=None)
    def defaultAction(edgeOrVertex, g, q, a):
        pass

    @overload(Edge, Graph, FIFOQueue, VertexList, return_type=None)
    def defaultAction(edgeOrVertex, g, q, a):
        pass

    @overload(Vertex, Graph, FIFOQueue, VertexList, return_type=None)
    def discoverVertex(v, g, q, a):
        a.mutate(cons(v, a, return_type=VertexList))

    @overload(InnerLoopContext, return_type=Graph)
    def first(p):
        return __pair.first(p)

    @overload(Graph, Vertex, return_type=InnerLoopContext)
    def makeInnerLoopContext(a, b):
        return __pair.makePair(a, b)

    @overload(Vertex, Graph, return_type=EdgeList)
    def outEdges(v, g):
        return __incidence_and_vertex_list_graph.outEdges(v, g)

    @overload(Graph, Vertex, return_type=None)
    def projectionBehaviorPair(a, b):
        pair = makeInnerLoopContext(a, b, return_type=InnerLoopContext)
        assert (first(pair, return_type=Graph)) == (a)
        assert (second(pair, return_type=Vertex)) == (b)

    @overload(Graph, return_type=VertexList)
    def vertices(g):
        return __incidence_and_vertex_list_graph.vertices(g)

    VertexCount = __incidence_and_vertex_list_graph.VertexCount
    @overload(Graph, return_type=VertexCount)
    def numVertices(g):
        return __incidence_and_vertex_list_graph.numVertices(g)

    @overload(Vertex, Graph, return_type=VertexCount)
    def outDegree(v, g):
        return __incidence_and_vertex_list_graph.outDegree(v, g)

    Color = __color_marker.Color
    __read_write_property_map = read_write_property_map(Vertex, VertexList, Color, cons, emptyVertexList, head, isEmpty, tail)
    ColorPropertyMap = __read_write_property_map.PropertyMap
    __triplet = triplet(VertexList, FIFOQueue, ColorPropertyMap)
    OuterLoopState = __triplet.Triplet
    __pair0 = pair(OuterLoopState, EdgeList)
    InnerLoopState = __pair0.Pair
    @overload(InnerLoopState, InnerLoopContext, return_type=bool)
    def bfsInnerLoopCond(state, ctx):
        edgeList = second(state, return_type=EdgeList)
        return not isEmpty(edgeList, return_type=bool)

    @overload(InnerLoopState, InnerLoopContext, return_type=None)
    def bfsInnerLoopRepeat(s, c):
        return __while_loop0.repeat(s, c)

    @overload(InnerLoopState, InnerLoopContext, return_type=None)
    def bfsInnerLoopStep(state, ctx):
        g = first(ctx, return_type=Graph)
        u = second(ctx, return_type=Vertex)
        outerState = first(state, return_type=OuterLoopState)
        x1 = first(outerState, return_type=VertexList)
        q1 = second(outerState, return_type=FIFOQueue)
        c1 = third(outerState, return_type=ColorPropertyMap)
        edgeList = second(state, return_type=EdgeList)
        e = head(edgeList, return_type=Edge)
        es = tail(edgeList, return_type=EdgeList)
        v = tgt(e, return_type=Vertex)
        defaultAction(e, g, q1, x1, return_type=None)
        vc = get(c1, v, return_type=Color)
        if (vc) == (white(return_type=Color)):
            if True:
                defaultAction(e, g, q1, x1, return_type=None)
                c2 = put(c1, v, gray(return_type=Color), return_type=ColorPropertyMap)
                discoverVertex(v, g, q1, x1, return_type=None)
                state.mutate(makeInnerLoopState(makeOuterLoopState(x1, push(v, q1, return_type=FIFOQueue), c2, return_type=OuterLoopState), es, return_type=InnerLoopState))
        else:
            if (vc) == (gray(return_type=Color)):
                if True:
                    defaultAction(e, g, q1, x1, return_type=None)
                    state.mutate(makeInnerLoopState(makeOuterLoopState(x1, q1, c1, return_type=OuterLoopState), es, return_type=InnerLoopState))
            else:
                if True:
                    defaultAction(e, g, q1, x1, return_type=None)
                    c2 = put(c1, u, black(return_type=Color), return_type=ColorPropertyMap)
                    state.mutate(makeInnerLoopState(makeOuterLoopState(x1, q1, c2, return_type=OuterLoopState), es, return_type=InnerLoopState))

    __while_loop0 = while_loop(InnerLoopContext, InnerLoopState, bfsInnerLoopCond, bfsInnerLoopStep)
    @overload(InnerLoopState, return_type=EdgeList)
    def second(p):
        return __pair0.second(p)

    @overload(InnerLoopState, InnerLoopContext, return_type=None)
    def whileLoopBehavior(s, c):
        mutableState = s
        if bfsInnerLoopCond(s, c, return_type=bool):
            if True:
                mutableState1 = s
                mutableState2 = s
                bfsInnerLoopRepeat(mutableState1, c, return_type=None)
                bfsInnerLoopStep(mutableState2, c, return_type=None)
                assert (mutableState1) == (mutableState2)
        else:
            if True:
                mutableState1 = s
                bfsInnerLoopRepeat(mutableState1, c, return_type=None)
                assert (mutableState1) == (s)

    @overload(OuterLoopState, Graph, return_type=bool)
    def bfsOuterLoopCond(state, g):
        q = second(state, return_type=FIFOQueue)
        return not isEmptyQueue(q, return_type=bool)

    @overload(OuterLoopState, Graph, return_type=None)
    def bfsOuterLoopRepeat(s, c):
        return __while_loop.repeat(s, c)

    @overload(OuterLoopState, Graph, return_type=None)
    def bfsOuterLoopStep(state, g):
        x = first(state, return_type=VertexList)
        q1 = second(state, return_type=FIFOQueue)
        c = third(state, return_type=ColorPropertyMap)
        u = front(q1, return_type=Vertex)
        q2 = pop(q1, return_type=FIFOQueue)
        defaultAction(u, g, q2, x, return_type=None)
        innerState = makeInnerLoopState(makeOuterLoopState(x, q2, c, return_type=OuterLoopState), outEdges(u, g, return_type=EdgeList), return_type=InnerLoopState)
        innerContext = makeInnerLoopContext(g, u, return_type=InnerLoopContext)
        bfsInnerLoopRepeat(innerState, innerContext, return_type=None)
        outerLoopStateAfterInnerLoop = first(innerState, return_type=OuterLoopState)
        x_end = first(outerLoopStateAfterInnerLoop, return_type=VertexList)
        q_end = second(outerLoopStateAfterInnerLoop, return_type=FIFOQueue)
        c_end = third(outerLoopStateAfterInnerLoop, return_type=ColorPropertyMap)
        defaultAction(u, g, q_end, x_end, return_type=None)
        state.mutate(makeOuterLoopState(x_end, q_end, c_end, return_type=OuterLoopState))

    __while_loop = while_loop(Graph, OuterLoopState, bfsOuterLoopCond, bfsOuterLoopStep)
    @overload(InnerLoopState, return_type=OuterLoopState)
    def first(p):
        return __pair0.first(p)

    @overload(OuterLoopState, return_type=VertexList)
    def first(p):
        return __triplet.first(p)

    @overload(OuterLoopState, EdgeList, return_type=InnerLoopState)
    def makeInnerLoopState(a, b):
        return __pair0.makePair(a, b)

    @overload(OuterLoopState, EdgeList, return_type=None)
    def projectionBehaviorPair(a, b):
        pair = makeInnerLoopState(a, b, return_type=InnerLoopState)
        assert (first(pair, return_type=OuterLoopState)) == (a)
        assert (second(pair, return_type=EdgeList)) == (b)

    @overload(OuterLoopState, return_type=FIFOQueue)
    def second(p):
        return __triplet.second(p)

    @overload(OuterLoopState, Graph, return_type=None)
    def whileLoopBehavior(s, c):
        mutableState = s
        if bfsOuterLoopCond(s, c, return_type=bool):
            if True:
                mutableState1 = s
                mutableState2 = s
                bfsOuterLoopRepeat(mutableState1, c, return_type=None)
                bfsOuterLoopStep(mutableState2, c, return_type=None)
                assert (mutableState1) == (mutableState2)
        else:
            if True:
                mutableState1 = s
                bfsOuterLoopRepeat(mutableState1, c, return_type=None)
                assert (mutableState1) == (s)

    @overload(Graph, Vertex, VertexList, FIFOQueue, ColorPropertyMap, return_type=None)
    def breadthFirstVisit(g, s, a, q, c):
        discoverVertex(s, g, q, a, return_type=None)
        q1 = push(s, q, return_type=FIFOQueue)
        c1 = put(c, s, gray(return_type=Color), return_type=ColorPropertyMap)
        outerState = makeOuterLoopState(a, q1, c1, return_type=OuterLoopState)
        bfsOuterLoopRepeat(outerState, g, return_type=None)
        a.mutate(first(outerState, return_type=VertexList))

    @overload(return_type=ColorPropertyMap)
    def emptyMap():
        return __read_write_property_map.emptyMap()

    @overload(VertexList, FIFOQueue, ColorPropertyMap, return_type=OuterLoopState)
    def makeOuterLoopState(a, b, c):
        return __triplet.makeTriplet(a, b, c)

    @overload(VertexList, FIFOQueue, ColorPropertyMap, return_type=None)
    def projectionBehaviorTriplet(a, b, c):
        triplet = makeOuterLoopState(a, b, c, return_type=OuterLoopState)
        assert (first(triplet, return_type=VertexList)) == (a)
        assert (second(triplet, return_type=FIFOQueue)) == (b)
        assert (third(triplet, return_type=ColorPropertyMap)) == (c)

    @overload(OuterLoopState, return_type=ColorPropertyMap)
    def third(p):
        return __triplet.third(p)

    @overload(return_type=Color)
    def black():
        return __color_marker.black()

    @overload(ColorPropertyMap, Vertex, return_type=Color)
    def get(pm, k):
        return __read_write_property_map.get(pm, k)

    @overload(return_type=Color)
    def gray():
        return __color_marker.gray()

    @overload(VertexList, Color, return_type=ColorPropertyMap)
    def initMap(kl, v):
        return __read_write_property_map.initMap(kl, v)

    @overload(ColorPropertyMap, Vertex, Color, return_type=ColorPropertyMap)
    def put(pm, k, v):
        return __read_write_property_map.put(pm, k, v)

    @overload(return_type=Color)
    def white():
        return __color_marker.white()

    __namedtuple = namedtuple("PyBFSTestVisitor", ["bfsInnerLoopCond", "bfsOuterLoopCond", "black", "breadthFirstSearch", "cons", "empty", "emptyEdgeList", "emptyMap", "emptyVertexList", "first", "front", "get", "gray", "head", "initMap", "isEmpty", "isEmptyQueue", "makeEdge", "makeInnerLoopContext", "makeInnerLoopState", "makeOuterLoopState", "numVertices", "outDegree", "outEdges", "pop", "projectionBehaviorPair", "projectionBehaviorTriplet", "push", "put", "second", "src", "tail", "tgt", "third", "vertices", "whileLoopBehavior", "white", "bfsInnerLoopRepeat", "bfsInnerLoopStep", "bfsOuterLoopRepeat", "bfsOuterLoopStep", "breadthFirstVisit", "defaultAction", "discoverVertex", "Color", "ColorPropertyMap", "Edge", "EdgeList", "FIFOQueue", "Graph", "InnerLoopContext", "InnerLoopState", "Int", "OuterLoopState", "Vertex", "VertexCount", "VertexList"])
    return __namedtuple(bfsInnerLoopCond, bfsOuterLoopCond, black, breadthFirstSearch, cons, empty, emptyEdgeList, emptyMap, emptyVertexList, first, front, get, gray, head, initMap, isEmpty, isEmptyQueue, makeEdge, makeInnerLoopContext, makeInnerLoopState, makeOuterLoopState, numVertices, outDegree, outEdges, pop, projectionBehaviorPair, projectionBehaviorTriplet, push, put, second, src, tail, tgt, third, vertices, whileLoopBehavior, white, bfsInnerLoopRepeat, bfsInnerLoopStep, bfsOuterLoopRepeat, bfsOuterLoopStep, breadthFirstVisit, defaultAction, discoverVertex, Color, ColorPropertyMap, Edge, EdgeList, FIFOQueue, Graph, InnerLoopContext, InnerLoopState, Int, OuterLoopState, Vertex, VertexCount, VertexList)


def PyDijkstraVisitor():
    overload = functools.partial(multiple_dispatch.overload, {})
    __color_marker = color_marker()
    __base_types = base_types()
    __base_float_ops = base_float_ops()
    Vertex = __base_types.Vertex
    __edge = edge(Vertex)
    __list_py0 = list_py(Vertex)
    VertexList = __list_py0.List
    @overload(return_type=VertexList)
    def emptyVertexList():
        return __list_py0.empty()

    @overload(VertexList, return_type=bool)
    def isEmpty(l):
        return __list_py0.isEmpty(l)

    @overload(VertexList, return_type=VertexList)
    def tail(l):
        return __list_py0.tail(l)

    __pair1 = pair(Vertex, Vertex)
    VertexPair = __pair1.Pair
    __list_py1 = list_py(VertexPair)
    VertexPairList = __list_py1.List
    @overload(return_type=VertexPairList)
    def emptyVertexPairList():
        return __list_py1.empty()

    @overload(VertexPairList, return_type=bool)
    def isEmpty(l):
        return __list_py1.isEmpty(l)

    @overload(VertexPairList, return_type=VertexPairList)
    def tail(l):
        return __list_py1.tail(l)

    @overload(VertexPair, VertexPairList, return_type=VertexPairList)
    def cons(a, l):
        return __list_py1.cons(a, l)

    @overload(VertexPairList, return_type=VertexPair)
    def head(l):
        return __list_py1.head(l)

    @overload(Vertex, VertexList, return_type=VertexList)
    def cons(a, l):
        return __list_py0.cons(a, l)

    @overload(VertexPair, return_type=Vertex)
    def first(p):
        return __pair1.first(p)

    @overload(VertexList, return_type=Vertex)
    def head(l):
        return __list_py0.head(l)

    @overload(Vertex, Vertex, return_type=VertexPair)
    def makeVertexPair(a, b):
        return __pair1.makePair(a, b)

    @overload(VertexPair, return_type=Vertex)
    def second(p):
        return __pair1.second(p)

    Int = __base_types.Int
    Edge = __edge.Edge
    __list_py = list_py(Edge)
    EdgeList = __list_py.List
    @overload(return_type=EdgeList)
    def emptyEdgeList():
        return __list_py.empty()

    @overload(EdgeList, return_type=bool)
    def isEmpty(l):
        return __list_py.isEmpty(l)

    @overload(EdgeList, return_type=EdgeList)
    def tail(l):
        return __list_py.tail(l)

    @overload(Edge, EdgeList, return_type=EdgeList)
    def cons(a, l):
        return __list_py.cons(a, l)

    __read_write_property_map2 = read_write_property_map(Vertex, VertexList, Vertex, cons, emptyVertexList, head, isEmpty, tail)
    VertexPredecessorMap = __read_write_property_map2.PropertyMap
    __pair2 = pair(VertexPredecessorMap, VertexList)
    PopulateVPMapState = __pair2.Pair
    @overload(PopulateVPMapState, Vertex, return_type=bool)
    def populateVPMapLoopCond(state, s):
        return not isEmpty(second(state, return_type=VertexList), return_type=bool)

    @overload(PopulateVPMapState, Vertex, return_type=None)
    def populateVPMapLoopRepeat(s, c):
        return __while_loop1.repeat(s, c)

    @overload(PopulateVPMapState, Vertex, return_type=None)
    def populateVPMapLoopStep(state, s):
        vpm = first(state, return_type=VertexPredecessorMap)
        vertexList = second(state, return_type=VertexList)
        v = head(vertexList, return_type=Vertex)
        state.mutate(makePair(put(vpm, v, v, return_type=VertexPredecessorMap), tail(vertexList, return_type=VertexList), return_type=PopulateVPMapState))

    __while_loop1 = while_loop(Vertex, PopulateVPMapState, populateVPMapLoopCond, populateVPMapLoopStep)
    @overload(PopulateVPMapState, return_type=VertexList)
    def second(p):
        return __pair2.second(p)

    @overload(PopulateVPMapState, Vertex, return_type=None)
    def whileLoopBehavior(s, c):
        mutableState = s
        if populateVPMapLoopCond(s, c, return_type=bool):
            if True:
                mutableState1 = s
                mutableState2 = s
                populateVPMapLoopRepeat(mutableState1, c, return_type=None)
                populateVPMapLoopStep(mutableState2, c, return_type=None)
                assert (mutableState1) == (mutableState2)
        else:
            if True:
                mutableState1 = s
                populateVPMapLoopRepeat(mutableState1, c, return_type=None)
                assert (mutableState1) == (s)

    @overload(return_type=VertexPredecessorMap)
    def emptyVPMap():
        return __read_write_property_map2.emptyMap()

    @overload(PopulateVPMapState, return_type=VertexPredecessorMap)
    def first(p):
        return __pair2.first(p)

    @overload(VertexPredecessorMap, Vertex, return_type=Vertex)
    def get(pm, k):
        return __read_write_property_map2.get(pm, k)

    @overload(VertexList, Vertex, return_type=VertexPredecessorMap)
    def initMap(kl, v):
        return __read_write_property_map2.initMap(kl, v)

    @overload(VertexPredecessorMap, VertexList, return_type=PopulateVPMapState)
    def makePair(a, b):
        return __pair2.makePair(a, b)

    @overload(VertexPredecessorMap, VertexList, return_type=None)
    def projectionBehaviorPair(a, b):
        pair = makePair(a, b, return_type=PopulateVPMapState)
        assert (first(pair, return_type=VertexPredecessorMap)) == (a)
        assert (second(pair, return_type=VertexList)) == (b)

    @overload(VertexPredecessorMap, Vertex, Vertex, return_type=VertexPredecessorMap)
    def put(pm, k, v):
        return __read_write_property_map2.put(pm, k, v)

    @overload(EdgeList, return_type=Edge)
    def head(l):
        return __list_py.head(l)

    @overload(Vertex, Vertex, return_type=Edge)
    def makeEdge(s, t):
        return __edge.makeEdge(s, t)

    @overload(Edge, return_type=Vertex)
    def src(e):
        return __edge.src(e)

    @overload(Edge, return_type=Vertex)
    def tgt(e):
        return __edge.tgt(e)

    __incidence_and_vertex_list_graph = incidence_and_vertex_list_graph(Edge, EdgeList, Vertex, VertexList, cons, cons, emptyEdgeList, emptyVertexList, head, head, isEmpty, isEmpty, makeEdge, src, tail, tail, tgt)
    Graph = __incidence_and_vertex_list_graph.Graph
    __pair = pair(Graph, Vertex)
    InnerLoopContext = __pair.Pair
    @overload(InnerLoopContext, return_type=Vertex)
    def second(p):
        return __pair.second(p)

    @overload(InnerLoopContext, return_type=Graph)
    def first(p):
        return __pair.first(p)

    @overload(Graph, Vertex, return_type=InnerLoopContext)
    def makeInnerLoopContext(a, b):
        return __pair.makePair(a, b)

    @overload(Vertex, Graph, return_type=EdgeList)
    def outEdges(v, g):
        return __incidence_and_vertex_list_graph.outEdges(v, g)

    @overload(Graph, Vertex, return_type=None)
    def projectionBehaviorPair(a, b):
        pair = makeInnerLoopContext(a, b, return_type=InnerLoopContext)
        assert (first(pair, return_type=Graph)) == (a)
        assert (second(pair, return_type=Vertex)) == (b)

    @overload(Graph, return_type=VertexList)
    def vertices(g):
        return __incidence_and_vertex_list_graph.vertices(g)

    VertexCount = __incidence_and_vertex_list_graph.VertexCount
    @overload(Graph, return_type=VertexCount)
    def numVertices(g):
        return __incidence_and_vertex_list_graph.numVertices(g)

    @overload(Vertex, Graph, return_type=VertexCount)
    def outDegree(v, g):
        return __incidence_and_vertex_list_graph.outDegree(v, g)

    Cost = __base_float_ops.Float
    __read_write_property_map = read_write_property_map(Edge, EdgeList, Cost, cons, emptyEdgeList, head, isEmpty, tail)
    EdgeCostMap = __read_write_property_map.PropertyMap
    @overload(return_type=EdgeCostMap)
    def emptyECMap():
        return __read_write_property_map.emptyMap()

    __read_write_property_map1 = read_write_property_map(Vertex, VertexList, Cost, cons, emptyVertexList, head, isEmpty, tail)
    VertexCostMap = __read_write_property_map1.PropertyMap
    __triplet0 = triplet(VertexCostMap, VertexPredecessorMap, EdgeCostMap)
    StateWithMaps = __triplet0.Triplet
    @overload(StateWithMaps, return_type=EdgeCostMap)
    def getEdgeCostMap(p):
        return __triplet0.third(p)

    @overload(StateWithMaps, return_type=VertexPredecessorMap)
    def getVertexPredecessorMap(p):
        return __triplet0.second(p)

    @overload(VertexPredecessorMap, StateWithMaps, return_type=StateWithMaps)
    def putVertexPredecessorMap(vpm, swm):
        return makeStateWithMaps(getVertexCostMap(swm, return_type=VertexCostMap), vpm, getEdgeCostMap(swm, return_type=EdgeCostMap), return_type=StateWithMaps)

    @overload(return_type=VertexCostMap)
    def emptyVCMap():
        return __read_write_property_map1.emptyMap()

    @overload(StateWithMaps, return_type=VertexCostMap)
    def getVertexCostMap(p):
        return __triplet0.first(p)

    @overload(VertexCostMap, VertexPredecessorMap, EdgeCostMap, return_type=StateWithMaps)
    def makeStateWithMaps(a, b, c):
        return __triplet0.makeTriplet(a, b, c)

    @overload(VertexCostMap, StateWithMaps, return_type=StateWithMaps)
    def putVertexCostMap(vcm, swm):
        return makeStateWithMaps(vcm, getVertexPredecessorMap(swm, return_type=VertexPredecessorMap), getEdgeCostMap(swm, return_type=EdgeCostMap), return_type=StateWithMaps)

    @overload(Edge, EdgeCostMap, VertexCostMap, VertexPredecessorMap, return_type=None)
    def relax(e, ecm, vcm, vpm):
        u = src(e, return_type=Vertex)
        v = tgt(e, return_type=Vertex)
        uCost = get(vcm, u, return_type=Cost)
        vCost = get(vcm, v, return_type=Cost)
        edgeCost = get(ecm, e, return_type=Cost)
        if less(plus(uCost, edgeCost, return_type=Cost), vCost, return_type=bool):
            if True:
                vcm.mutate(put(vcm, v, plus(uCost, edgeCost, return_type=Cost), return_type=VertexCostMap))
                vpm.mutate(put(vpm, v, u, return_type=VertexPredecessorMap))
        else:
            pass

    @overload(Graph, Vertex, VertexCostMap, EdgeCostMap, Cost, VertexPredecessorMap, return_type=None)
    def dijkstraShortestPaths(g, start, vcm, ecm, initialCost, vpm):
        vcm.mutate(put(vcm, start, initialCost, return_type=VertexCostMap))
        populateVPMapState = makePair(emptyVPMap(return_type=VertexPredecessorMap), vertices(g, return_type=VertexList), return_type=PopulateVPMapState)
        populateVPMapLoopRepeat(populateVPMapState, start, return_type=None)
        vpm.mutate(first(populateVPMapState, return_type=VertexPredecessorMap))
        pq = emptyPriorityQueue(vcm, return_type=PriorityQueue)
        swm = makeStateWithMaps(vcm, vpm, ecm, return_type=StateWithMaps)
        c = initMap(vertices(g, return_type=VertexList), white(return_type=Color), return_type=ColorPropertyMap)
        breadthFirstVisit(g, start, swm, pq, c, return_type=None)
        vcm.mutate(getVertexCostMap(swm, return_type=VertexCostMap))
        vpm.mutate(getVertexPredecessorMap(swm, return_type=VertexPredecessorMap))

    @overload(VertexCostMap, Vertex, return_type=Cost)
    def get(pm, k):
        return __read_write_property_map1.get(pm, k)

    __priority_queue = priority_queue(Vertex, Cost, VertexCostMap, get)
    PriorityQueue = __priority_queue.PriorityQueue
    @overload(Edge, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def blackTarget(edgeOrVertex, g, q, a):
        pass

    @overload(Vertex, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def discoverVertex(edgeOrVertex, g, q, a):
        pass

    @overload(VertexCostMap, return_type=PriorityQueue)
    def emptyPriorityQueue(pm):
        return __priority_queue.empty(pm)

    @overload(Edge, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def examineEdge(edgeOrVertex, g, q, a):
        pass

    @overload(Vertex, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def examineVertex(edgeOrVertex, g, q, a):
        pass

    @overload(Vertex, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def finishVertex(edgeOrVertex, g, q, a):
        pass

    @overload(PriorityQueue, return_type=Vertex)
    def front(q):
        return __priority_queue.front(q)

    @overload(Edge, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def grayTarget(e, g, pq, swm):
        origVcm = getVertexCostMap(swm, return_type=VertexCostMap)
        vpm = getVertexPredecessorMap(swm, return_type=VertexPredecessorMap)
        ecm = getEdgeCostMap(swm, return_type=EdgeCostMap)
        vcm = origVcm
        relax(e, ecm, vcm, vpm, return_type=None)
        if (vcm) == (origVcm):
            pass
        else:
            if True:
                swm.mutate(putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm, return_type=StateWithMaps), return_type=StateWithMaps))
                pq.mutate(update(vcm, tgt(e, return_type=Vertex), pq, return_type=PriorityQueue))

    @overload(PriorityQueue, return_type=bool)
    def isEmptyQueue(q):
        return __priority_queue.isEmpty(q)

    @overload(Edge, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def nonTreeEdge(edgeOrVertex, g, q, a):
        pass

    @overload(PriorityQueue, return_type=PriorityQueue)
    def pop(q):
        return __priority_queue.pop(q)

    @overload(Vertex, PriorityQueue, return_type=PriorityQueue)
    def push(a, q):
        return __priority_queue.push(a, q)

    @overload(Edge, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def treeEdge(e, g, pq, swm):
        vcm = getVertexCostMap(swm, return_type=VertexCostMap)
        vpm = getVertexPredecessorMap(swm, return_type=VertexPredecessorMap)
        ecm = getEdgeCostMap(swm, return_type=EdgeCostMap)
        relax(e, ecm, vcm, vpm, return_type=None)
        swm.mutate(putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm, return_type=StateWithMaps), return_type=StateWithMaps))

    @overload(VertexCostMap, Vertex, PriorityQueue, return_type=PriorityQueue)
    def update(pm, a, pq):
        return __priority_queue.update(pm, a, pq)

    @overload(EdgeCostMap, Edge, return_type=Cost)
    def get(pm, k):
        return __read_write_property_map.get(pm, k)

    @overload(VertexList, Cost, return_type=VertexCostMap)
    def initMap(kl, v):
        return __read_write_property_map1.initMap(kl, v)

    @overload(EdgeList, Cost, return_type=EdgeCostMap)
    def initMap(kl, v):
        return __read_write_property_map.initMap(kl, v)

    @overload(Cost, Cost, return_type=bool)
    def less(i1, i2):
        return __base_float_ops.less(i1, i2)

    @overload(Cost, Cost, return_type=Cost)
    def plus(i1, i2):
        return __base_float_ops.plus(i1, i2)

    @overload(VertexCostMap, Vertex, Cost, return_type=VertexCostMap)
    def put(pm, k, v):
        return __read_write_property_map1.put(pm, k, v)

    @overload(EdgeCostMap, Edge, Cost, return_type=EdgeCostMap)
    def put(pm, k, v):
        return __read_write_property_map.put(pm, k, v)

    Color = __color_marker.Color
    __read_write_property_map0 = read_write_property_map(Vertex, VertexList, Color, cons, emptyVertexList, head, isEmpty, tail)
    ColorPropertyMap = __read_write_property_map0.PropertyMap
    __triplet = triplet(StateWithMaps, PriorityQueue, ColorPropertyMap)
    OuterLoopState = __triplet.Triplet
    __pair0 = pair(OuterLoopState, EdgeList)
    InnerLoopState = __pair0.Pair
    @overload(InnerLoopState, InnerLoopContext, return_type=bool)
    def bfsInnerLoopCond(state, ctx):
        edgeList = second(state, return_type=EdgeList)
        return not isEmpty(edgeList, return_type=bool)

    @overload(InnerLoopState, InnerLoopContext, return_type=None)
    def bfsInnerLoopRepeat(s, c):
        return __while_loop0.repeat(s, c)

    @overload(InnerLoopState, InnerLoopContext, return_type=None)
    def bfsInnerLoopStep(state, ctx):
        g = first(ctx, return_type=Graph)
        u = second(ctx, return_type=Vertex)
        outerState = first(state, return_type=OuterLoopState)
        x1 = first(outerState, return_type=StateWithMaps)
        q1 = second(outerState, return_type=PriorityQueue)
        c1 = third(outerState, return_type=ColorPropertyMap)
        edgeList = second(state, return_type=EdgeList)
        e = head(edgeList, return_type=Edge)
        es = tail(edgeList, return_type=EdgeList)
        v = tgt(e, return_type=Vertex)
        examineEdge(e, g, q1, x1, return_type=None)
        vc = get(c1, v, return_type=Color)
        if (vc) == (white(return_type=Color)):
            if True:
                treeEdge(e, g, q1, x1, return_type=None)
                c2 = put(c1, v, gray(return_type=Color), return_type=ColorPropertyMap)
                discoverVertex(v, g, q1, x1, return_type=None)
                state.mutate(makeInnerLoopState(makeOuterLoopState(x1, push(v, q1, return_type=PriorityQueue), c2, return_type=OuterLoopState), es, return_type=InnerLoopState))
        else:
            if (vc) == (gray(return_type=Color)):
                if True:
                    grayTarget(e, g, q1, x1, return_type=None)
                    state.mutate(makeInnerLoopState(makeOuterLoopState(x1, q1, c1, return_type=OuterLoopState), es, return_type=InnerLoopState))
            else:
                if True:
                    blackTarget(e, g, q1, x1, return_type=None)
                    c2 = put(c1, u, black(return_type=Color), return_type=ColorPropertyMap)
                    state.mutate(makeInnerLoopState(makeOuterLoopState(x1, q1, c2, return_type=OuterLoopState), es, return_type=InnerLoopState))

    __while_loop0 = while_loop(InnerLoopContext, InnerLoopState, bfsInnerLoopCond, bfsInnerLoopStep)
    @overload(InnerLoopState, return_type=EdgeList)
    def second(p):
        return __pair0.second(p)

    @overload(InnerLoopState, InnerLoopContext, return_type=None)
    def whileLoopBehavior(s, c):
        mutableState = s
        if bfsInnerLoopCond(s, c, return_type=bool):
            if True:
                mutableState1 = s
                mutableState2 = s
                bfsInnerLoopRepeat(mutableState1, c, return_type=None)
                bfsInnerLoopStep(mutableState2, c, return_type=None)
                assert (mutableState1) == (mutableState2)
        else:
            if True:
                mutableState1 = s
                bfsInnerLoopRepeat(mutableState1, c, return_type=None)
                assert (mutableState1) == (s)

    @overload(OuterLoopState, Graph, return_type=bool)
    def bfsOuterLoopCond(state, g):
        q = second(state, return_type=PriorityQueue)
        return not isEmptyQueue(q, return_type=bool)

    @overload(OuterLoopState, Graph, return_type=None)
    def bfsOuterLoopRepeat(s, c):
        return __while_loop.repeat(s, c)

    @overload(OuterLoopState, Graph, return_type=None)
    def bfsOuterLoopStep(state, g):
        x = first(state, return_type=StateWithMaps)
        q1 = second(state, return_type=PriorityQueue)
        c = third(state, return_type=ColorPropertyMap)
        u = front(q1, return_type=Vertex)
        q2 = pop(q1, return_type=PriorityQueue)
        examineVertex(u, g, q2, x, return_type=None)
        innerState = makeInnerLoopState(makeOuterLoopState(x, q2, c, return_type=OuterLoopState), outEdges(u, g, return_type=EdgeList), return_type=InnerLoopState)
        innerContext = makeInnerLoopContext(g, u, return_type=InnerLoopContext)
        bfsInnerLoopRepeat(innerState, innerContext, return_type=None)
        outerLoopStateAfterInnerLoop = first(innerState, return_type=OuterLoopState)
        x_end = first(outerLoopStateAfterInnerLoop, return_type=StateWithMaps)
        q_end = second(outerLoopStateAfterInnerLoop, return_type=PriorityQueue)
        c_end = third(outerLoopStateAfterInnerLoop, return_type=ColorPropertyMap)
        finishVertex(u, g, q_end, x_end, return_type=None)
        state.mutate(makeOuterLoopState(x_end, q_end, c_end, return_type=OuterLoopState))

    __while_loop = while_loop(Graph, OuterLoopState, bfsOuterLoopCond, bfsOuterLoopStep)
    @overload(InnerLoopState, return_type=OuterLoopState)
    def first(p):
        return __pair0.first(p)

    @overload(OuterLoopState, return_type=StateWithMaps)
    def first(p):
        return __triplet.first(p)

    @overload(OuterLoopState, EdgeList, return_type=InnerLoopState)
    def makeInnerLoopState(a, b):
        return __pair0.makePair(a, b)

    @overload(OuterLoopState, EdgeList, return_type=None)
    def projectionBehaviorPair(a, b):
        pair = makeInnerLoopState(a, b, return_type=InnerLoopState)
        assert (first(pair, return_type=OuterLoopState)) == (a)
        assert (second(pair, return_type=EdgeList)) == (b)

    @overload(OuterLoopState, return_type=PriorityQueue)
    def second(p):
        return __triplet.second(p)

    @overload(OuterLoopState, Graph, return_type=None)
    def whileLoopBehavior(s, c):
        mutableState = s
        if bfsOuterLoopCond(s, c, return_type=bool):
            if True:
                mutableState1 = s
                mutableState2 = s
                bfsOuterLoopRepeat(mutableState1, c, return_type=None)
                bfsOuterLoopStep(mutableState2, c, return_type=None)
                assert (mutableState1) == (mutableState2)
        else:
            if True:
                mutableState1 = s
                bfsOuterLoopRepeat(mutableState1, c, return_type=None)
                assert (mutableState1) == (s)

    @overload(Graph, Vertex, StateWithMaps, PriorityQueue, ColorPropertyMap, return_type=None)
    def breadthFirstVisit(g, s, a, q, c):
        discoverVertex(s, g, q, a, return_type=None)
        q1 = push(s, q, return_type=PriorityQueue)
        c1 = put(c, s, gray(return_type=Color), return_type=ColorPropertyMap)
        outerState = makeOuterLoopState(a, q1, c1, return_type=OuterLoopState)
        bfsOuterLoopRepeat(outerState, g, return_type=None)
        a.mutate(first(outerState, return_type=StateWithMaps))

    @overload(return_type=ColorPropertyMap)
    def emptyMap():
        return __read_write_property_map0.emptyMap()

    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, return_type=OuterLoopState)
    def makeOuterLoopState(a, b, c):
        return __triplet.makeTriplet(a, b, c)

    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, return_type=None)
    def projectionBehaviorTriplet(a, b, c):
        triplet = makeOuterLoopState(a, b, c, return_type=OuterLoopState)
        assert (first(triplet, return_type=StateWithMaps)) == (a)
        assert (second(triplet, return_type=PriorityQueue)) == (b)
        assert (third(triplet, return_type=ColorPropertyMap)) == (c)

    @overload(OuterLoopState, return_type=ColorPropertyMap)
    def third(p):
        return __triplet.third(p)

    @overload(return_type=Color)
    def black():
        return __color_marker.black()

    @overload(ColorPropertyMap, Vertex, return_type=Color)
    def get(pm, k):
        return __read_write_property_map0.get(pm, k)

    @overload(return_type=Color)
    def gray():
        return __color_marker.gray()

    @overload(VertexList, Color, return_type=ColorPropertyMap)
    def initMap(kl, v):
        return __read_write_property_map0.initMap(kl, v)

    @overload(ColorPropertyMap, Vertex, Color, return_type=ColorPropertyMap)
    def put(pm, k, v):
        return __read_write_property_map0.put(pm, k, v)

    @overload(return_type=Color)
    def white():
        return __color_marker.white()

    __namedtuple = namedtuple("PyDijkstraVisitor", ["bfsInnerLoopCond", "bfsOuterLoopCond", "black", "cons", "emptyECMap", "emptyEdgeList", "emptyMap", "emptyPriorityQueue", "emptyVCMap", "emptyVPMap", "emptyVertexList", "emptyVertexPairList", "first", "front", "get", "getEdgeCostMap", "getVertexCostMap", "getVertexPredecessorMap", "gray", "head", "initMap", "isEmpty", "isEmptyQueue", "less", "makeEdge", "makeInnerLoopContext", "makeInnerLoopState", "makeOuterLoopState", "makePair", "makeStateWithMaps", "makeVertexPair", "numVertices", "outDegree", "outEdges", "plus", "pop", "populateVPMapLoopCond", "projectionBehaviorPair", "projectionBehaviorTriplet", "push", "put", "putVertexCostMap", "putVertexPredecessorMap", "second", "src", "tail", "tgt", "third", "update", "vertices", "whileLoopBehavior", "white", "bfsInnerLoopRepeat", "bfsInnerLoopStep", "bfsOuterLoopRepeat", "bfsOuterLoopStep", "blackTarget", "breadthFirstVisit", "dijkstraShortestPaths", "discoverVertex", "examineEdge", "examineVertex", "finishVertex", "grayTarget", "nonTreeEdge", "populateVPMapLoopRepeat", "populateVPMapLoopStep", "relax", "treeEdge", "Color", "ColorPropertyMap", "Cost", "Edge", "EdgeCostMap", "EdgeList", "Graph", "InnerLoopContext", "InnerLoopState", "Int", "OuterLoopState", "PopulateVPMapState", "PriorityQueue", "StateWithMaps", "Vertex", "VertexCostMap", "VertexCount", "VertexList", "VertexPair", "VertexPairList", "VertexPredecessorMap"])
    return __namedtuple(bfsInnerLoopCond, bfsOuterLoopCond, black, cons, emptyECMap, emptyEdgeList, emptyMap, emptyPriorityQueue, emptyVCMap, emptyVPMap, emptyVertexList, emptyVertexPairList, first, front, get, getEdgeCostMap, getVertexCostMap, getVertexPredecessorMap, gray, head, initMap, isEmpty, isEmptyQueue, less, makeEdge, makeInnerLoopContext, makeInnerLoopState, makeOuterLoopState, makePair, makeStateWithMaps, makeVertexPair, numVertices, outDegree, outEdges, plus, pop, populateVPMapLoopCond, projectionBehaviorPair, projectionBehaviorTriplet, push, put, putVertexCostMap, putVertexPredecessorMap, second, src, tail, tgt, third, update, vertices, whileLoopBehavior, white, bfsInnerLoopRepeat, bfsInnerLoopStep, bfsOuterLoopRepeat, bfsOuterLoopStep, blackTarget, breadthFirstVisit, dijkstraShortestPaths, discoverVertex, examineEdge, examineVertex, finishVertex, grayTarget, nonTreeEdge, populateVPMapLoopRepeat, populateVPMapLoopStep, relax, treeEdge, Color, ColorPropertyMap, Cost, Edge, EdgeCostMap, EdgeList, Graph, InnerLoopContext, InnerLoopState, Int, OuterLoopState, PopulateVPMapState, PriorityQueue, StateWithMaps, Vertex, VertexCostMap, VertexCount, VertexList, VertexPair, VertexPairList, VertexPredecessorMap)
