from base import base_float_ops
from base import base_types
from base import fifo_queue
from base import for_iterator_loop
from base import for_iterator_loop3_2
from base import for_parallel_iterator_loop3_2
from base import incidence_and_vertex_list_graph
from base import list_py
from base import pair
from base import priority_queue
from base import read_write_property_map
from base import stack
from base import triplet
from base import two_bit_color_map
from base import vector
from base import while_loop
from base import while_loop3
from collections import namedtuple
import functools
import multiple_dispatch


def PyBFSTestVisitor():
    overload = functools.partial(multiple_dispatch.overload, {})
    __base_types = base_types()
    Vertex = __base_types.Vertex
    __incidence_and_vertex_list_graph = incidence_and_vertex_list_graph(Vertex)
    VertexCount = __incidence_and_vertex_list_graph.VertexCount
    VertexDescriptor = __incidence_and_vertex_list_graph.VertexDescriptor
    __fifo_queue = fifo_queue(VertexDescriptor)
    __vector = vector(VertexDescriptor)
    VertexVector = __vector.Vector
    @overload(return_type=VertexVector)
    def emptyVertexVector():
        return __vector.empty()

    @overload(VertexDescriptor, VertexVector, return_type=None)
    def pushBack(a, v):
        return __vector.pushBack(a, v)

    VertexIterator = __incidence_and_vertex_list_graph.VertexIterator
    @overload(VertexIterator, return_type=bool)
    def vertexIterEnd(ei):
        return __incidence_and_vertex_list_graph.vertexIterEnd(ei)

    @overload(VertexIterator, return_type=None)
    def vertexIterNext(ei):
        return __incidence_and_vertex_list_graph.vertexIterNext(ei)

    @overload(VertexIterator, return_type=VertexDescriptor)
    def vertexIterUnpack(ei):
        return __incidence_and_vertex_list_graph.vertexIterUnpack(ei)

    __two_bit_color_map = two_bit_color_map(VertexDescriptor, VertexIterator, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    Int = __base_types.Int
    Graph = __incidence_and_vertex_list_graph.Graph
    @overload(Graph, VertexDescriptor, VertexVector, return_type=VertexVector)
    def breadthFirstSearch(g, start, init):
        q = empty(return_type=FIFOQueue)
        vertexItr = VertexIterator()
        vertices(g, vertexItr, return_type=None)
        c = initMap(vertexItr, white(return_type=Color), return_type=ColorPropertyMap)
        a = init
        breadthFirstVisit(g, start, a, q, c, return_type=None)
        return a

    @overload(Graph, return_type=VertexCount)
    def numVertices(g):
        return __incidence_and_vertex_list_graph.numVertices(g)

    @overload(VertexDescriptor, Graph, return_type=VertexCount)
    def outDegree(v, g):
        return __incidence_and_vertex_list_graph.outDegree(v, g)

    @overload(Vertex, Graph, return_type=VertexDescriptor)
    def toVertexDescriptor(v, g):
        return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g)

    @overload(Graph, VertexIterator, return_type=None)
    def vertices(g, itr):
        return __incidence_and_vertex_list_graph.vertices(g, itr)

    FIFOQueue = __fifo_queue.FIFOQueue
    @overload(VertexDescriptor, Graph, FIFOQueue, VertexVector, return_type=None)
    def defaultAction(edgeOrVertex, g, q, a):
        pass

    @overload(return_type=FIFOQueue)
    def empty():
        return __fifo_queue.empty()

    @overload(VertexDescriptor, Graph, FIFOQueue, VertexVector, return_type=None)
    def examineVertex(v, g, q, a):
        pushBack(v, a, return_type=None)

    @overload(FIFOQueue, return_type=VertexDescriptor)
    def front(q):
        return __fifo_queue.front(q)

    @overload(FIFOQueue, return_type=bool)
    def isEmptyQueue(q):
        return __fifo_queue.isEmpty(q)

    @overload(FIFOQueue, return_type=None)
    def pop(q):
        return __fifo_queue.pop(q)

    @overload(VertexDescriptor, FIFOQueue, return_type=None)
    def push(a, q):
        return __fifo_queue.push(a, q)

    @overload(VertexDescriptor, FIFOQueue, return_type=None)
    def pushPopBehavior(a, inq):
        mut_inq = inq
        push(a, mut_inq, return_type=None)
        assert (front(mut_inq, return_type=VertexDescriptor)) == (a)
        pop(mut_inq, return_type=None)
        assert (inq) == (mut_inq)

    EdgeIterator = __incidence_and_vertex_list_graph.EdgeIterator
    __pair = pair(EdgeIterator, EdgeIterator)
    EdgeIteratorRange = __pair.Pair
    @overload(EdgeIterator, return_type=bool)
    def edgeIterEnd(ei):
        return __incidence_and_vertex_list_graph.edgeIterEnd(ei)

    @overload(EdgeIterator, return_type=None)
    def edgeIterNext(ei):
        return __incidence_and_vertex_list_graph.edgeIterNext(ei)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeBegin(p):
        return __pair.first(p)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeEnd(p):
        return __pair.second(p)

    @overload(EdgeIterator, EdgeIterator, return_type=EdgeIteratorRange)
    def makeEdgeIteratorRange(a, b):
        return __pair.makePair(a, b)

    @overload(VertexDescriptor, Graph, EdgeIterator, return_type=None)
    def outEdges(v, g, itr):
        return __incidence_and_vertex_list_graph.outEdges(v, g, itr)

    EdgeDescriptor = __incidence_and_vertex_list_graph.EdgeDescriptor
    @overload(EdgeDescriptor, Graph, FIFOQueue, VertexVector, return_type=None)
    def defaultAction(edgeOrVertex, g, q, a):
        pass

    @overload(EdgeIterator, return_type=EdgeDescriptor)
    def edgeIterUnpack(ei):
        return __incidence_and_vertex_list_graph.edgeIterUnpack(ei)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def src(e, g):
        return __incidence_and_vertex_list_graph.src(e, g)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def tgt(e, g):
        return __incidence_and_vertex_list_graph.tgt(e, g)

    @overload(VertexDescriptor, VertexDescriptor, Graph, return_type=EdgeDescriptor)
    def toEdgeDescriptor(v1, v2, g):
        return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g)

    Edge = __incidence_and_vertex_list_graph.Edge
    @overload(Vertex, Vertex, return_type=Edge)
    def makeEdge(s, t):
        return __incidence_and_vertex_list_graph.makeEdge(s, t)

    ColorPropertyMap = __two_bit_color_map.ColorPropertyMap
    @overload(EdgeIterator, VertexVector, FIFOQueue, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopRepeat(itr, s1, s2, s3, ctx1, ctx2):
        return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2)

    @overload(EdgeIterator, VertexVector, FIFOQueue, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopStep(edgeItr, x, q, c, g, u):
        e = edgeIterUnpack(edgeItr, return_type=EdgeDescriptor)
        v = tgt(e, g, return_type=VertexDescriptor)
        defaultAction(e, g, q, x, return_type=None)
        vc = get(c, v, return_type=Color)
        if (vc) == (white(return_type=Color)):
            if True:
                defaultAction(e, g, q, x, return_type=None)
                put(c, v, gray(return_type=Color), return_type=None)
                defaultAction(v, g, q, x, return_type=None)
                push(v, q, return_type=None)
        else:
            if (vc) == (gray(return_type=Color)):
                if True:
                    defaultAction(e, g, q, x, return_type=None)
            else:
                if True:
                    defaultAction(e, g, q, x, return_type=None)

    __for_iterator_loop3_2 = for_iterator_loop3_2(Graph, VertexDescriptor, EdgeIterator, VertexVector, FIFOQueue, ColorPropertyMap, edgeIterEnd, edgeIterNext, bfsInnerLoopStep)
    @overload(VertexVector, FIFOQueue, ColorPropertyMap, Graph, return_type=bool)
    def bfsOuterLoopCond(a, q, c, g):
        return not isEmptyQueue(q, return_type=bool)

    @overload(VertexVector, FIFOQueue, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopRepeat(s1, s2, s3, ctx):
        return __while_loop3.repeat(s1, s2, s3, ctx)

    @overload(VertexVector, FIFOQueue, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopStep(x, q, c, g):
        u = front(q, return_type=VertexDescriptor)
        pop(q, return_type=None)
        examineVertex(u, g, q, x, return_type=None)
        edgeItr = EdgeIterator()
        outEdges(u, g, edgeItr, return_type=None)
        bfsInnerLoopRepeat(edgeItr, x, q, c, g, u, return_type=None)
        put(c, u, black(return_type=Color), return_type=None)
        defaultAction(u, g, q, x, return_type=None)

    __while_loop3 = while_loop3(Graph, VertexVector, FIFOQueue, ColorPropertyMap, bfsOuterLoopCond, bfsOuterLoopStep)
    @overload(Graph, VertexDescriptor, VertexVector, FIFOQueue, ColorPropertyMap, return_type=None)
    def breadthFirstVisit(g, s, a, q, c):
        defaultAction(s, g, q, a, return_type=None)
        push(s, q, return_type=None)
        put(c, s, gray(return_type=Color), return_type=None)
        bfsOuterLoopRepeat(a, q, c, g, return_type=None)

    Color = __two_bit_color_map.Color
    @overload(return_type=Color)
    def black():
        return __two_bit_color_map.black()

    @overload(ColorPropertyMap, VertexDescriptor, return_type=Color)
    def get(pm, k):
        return __two_bit_color_map.get(pm, k)

    @overload(return_type=Color)
    def gray():
        return __two_bit_color_map.gray()

    @overload(VertexIterator, Color, return_type=ColorPropertyMap)
    def initMap(kli, v):
        return __two_bit_color_map.initMap(kli, v)

    @overload(ColorPropertyMap, VertexDescriptor, Color, return_type=None)
    def put(pm, k, v):
        return __two_bit_color_map.put(pm, k, v)

    @overload(return_type=Color)
    def white():
        return __two_bit_color_map.white()

    __namedtuple = namedtuple("PyBFSTestVisitor", ["Color", "ColorPropertyMap", "Edge", "EdgeDescriptor", "EdgeIterator", "EdgeIteratorRange", "FIFOQueue", "Graph", "Int", "Vertex", "VertexCount", "VertexDescriptor", "VertexIterator", "VertexVector", "bfsInnerLoopRepeat", "bfsInnerLoopStep", "bfsOuterLoopCond", "bfsOuterLoopRepeat", "bfsOuterLoopStep", "black", "breadthFirstSearch", "breadthFirstVisit", "defaultAction", "edgeIterEnd", "edgeIterNext", "edgeIterUnpack", "empty", "emptyVertexVector", "examineVertex", "front", "get", "gray", "initMap", "isEmptyQueue", "iterRangeBegin", "iterRangeEnd", "makeEdge", "makeEdgeIteratorRange", "numVertices", "outDegree", "outEdges", "pop", "push", "pushBack", "pushPopBehavior", "put", "src", "tgt", "toEdgeDescriptor", "toVertexDescriptor", "vertexIterEnd", "vertexIterNext", "vertexIterUnpack", "vertices", "white"])
    return __namedtuple(Color, ColorPropertyMap, Edge, EdgeDescriptor, EdgeIterator, EdgeIteratorRange, FIFOQueue, Graph, Int, Vertex, VertexCount, VertexDescriptor, VertexIterator, VertexVector, bfsInnerLoopRepeat, bfsInnerLoopStep, bfsOuterLoopCond, bfsOuterLoopRepeat, bfsOuterLoopStep, black, breadthFirstSearch, breadthFirstVisit, defaultAction, edgeIterEnd, edgeIterNext, edgeIterUnpack, empty, emptyVertexVector, examineVertex, front, get, gray, initMap, isEmptyQueue, iterRangeBegin, iterRangeEnd, makeEdge, makeEdgeIteratorRange, numVertices, outDegree, outEdges, pop, push, pushBack, pushPopBehavior, put, src, tgt, toEdgeDescriptor, toVertexDescriptor, vertexIterEnd, vertexIterNext, vertexIterUnpack, vertices, white)


def PyDFSTestVisitor():
    overload = functools.partial(multiple_dispatch.overload, {})
    @overload(return_type=None)
    def emptyStackIsEmpty():
        assert isEmptyStack(empty(return_type=Stack), return_type=bool)

    __base_types = base_types()
    Vertex = __base_types.Vertex
    __incidence_and_vertex_list_graph = incidence_and_vertex_list_graph(Vertex)
    VertexCount = __incidence_and_vertex_list_graph.VertexCount
    VertexDescriptor = __incidence_and_vertex_list_graph.VertexDescriptor
    __stack = stack(VertexDescriptor)
    __vector = vector(VertexDescriptor)
    VertexVector = __vector.Vector
    @overload(return_type=VertexVector)
    def emptyVertexVector():
        return __vector.empty()

    @overload(VertexDescriptor, VertexVector, return_type=None)
    def pushBack(a, v):
        return __vector.pushBack(a, v)

    VertexIterator = __incidence_and_vertex_list_graph.VertexIterator
    @overload(VertexIterator, return_type=bool)
    def vertexIterEnd(ei):
        return __incidence_and_vertex_list_graph.vertexIterEnd(ei)

    @overload(VertexIterator, return_type=None)
    def vertexIterNext(ei):
        return __incidence_and_vertex_list_graph.vertexIterNext(ei)

    @overload(VertexIterator, return_type=VertexDescriptor)
    def vertexIterUnpack(ei):
        return __incidence_and_vertex_list_graph.vertexIterUnpack(ei)

    __two_bit_color_map = two_bit_color_map(VertexDescriptor, VertexIterator, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    Stack = __stack.Stack
    @overload(return_type=Stack)
    def empty():
        return __stack.empty()

    @overload(Stack, return_type=bool)
    def isEmptyStack(s):
        return __stack.isEmpty(s)

    @overload(Stack, return_type=None)
    def pop(s):
        return __stack.pop(s)

    @overload(VertexDescriptor, Stack, return_type=None)
    def push(a, s):
        return __stack.push(a, s)

    @overload(Stack, VertexDescriptor, return_type=None)
    def pushPopTopBehavior(s, a):
        mut_s = s
        push(a, mut_s, return_type=None)
        assert (top(mut_s, return_type=VertexDescriptor)) == (a)
        pop(mut_s, return_type=None)
        assert (mut_s) == (s)

    @overload(Stack, return_type=VertexDescriptor)
    def top(s):
        return __stack.top(s)

    Int = __base_types.Int
    Graph = __incidence_and_vertex_list_graph.Graph
    @overload(VertexDescriptor, Graph, Stack, VertexVector, return_type=None)
    def defaultAction(edgeOrVertex, g, q, a):
        pass

    @overload(Graph, VertexDescriptor, VertexVector, return_type=VertexVector)
    def depthFirstSearch(g, start, init):
        q = empty(return_type=Stack)
        vertexItr = VertexIterator()
        vertices(g, vertexItr, return_type=None)
        c = initMap(vertexItr, white(return_type=Color), return_type=ColorPropertyMap)
        a = init
        breadthFirstVisit(g, start, a, q, c, return_type=None)
        return a

    @overload(VertexDescriptor, Graph, Stack, VertexVector, return_type=None)
    def examineVertex(v, g, q, a):
        pushBack(v, a, return_type=None)

    @overload(Graph, return_type=VertexCount)
    def numVertices(g):
        return __incidence_and_vertex_list_graph.numVertices(g)

    @overload(VertexDescriptor, Graph, return_type=VertexCount)
    def outDegree(v, g):
        return __incidence_and_vertex_list_graph.outDegree(v, g)

    @overload(Vertex, Graph, return_type=VertexDescriptor)
    def toVertexDescriptor(v, g):
        return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g)

    @overload(Graph, VertexIterator, return_type=None)
    def vertices(g, itr):
        return __incidence_and_vertex_list_graph.vertices(g, itr)

    EdgeIterator = __incidence_and_vertex_list_graph.EdgeIterator
    __pair = pair(EdgeIterator, EdgeIterator)
    EdgeIteratorRange = __pair.Pair
    @overload(EdgeIterator, return_type=bool)
    def edgeIterEnd(ei):
        return __incidence_and_vertex_list_graph.edgeIterEnd(ei)

    @overload(EdgeIterator, return_type=None)
    def edgeIterNext(ei):
        return __incidence_and_vertex_list_graph.edgeIterNext(ei)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeBegin(p):
        return __pair.first(p)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeEnd(p):
        return __pair.second(p)

    @overload(EdgeIterator, EdgeIterator, return_type=EdgeIteratorRange)
    def makeEdgeIteratorRange(a, b):
        return __pair.makePair(a, b)

    @overload(VertexDescriptor, Graph, EdgeIterator, return_type=None)
    def outEdges(v, g, itr):
        return __incidence_and_vertex_list_graph.outEdges(v, g, itr)

    EdgeDescriptor = __incidence_and_vertex_list_graph.EdgeDescriptor
    @overload(EdgeDescriptor, Graph, Stack, VertexVector, return_type=None)
    def defaultAction(edgeOrVertex, g, q, a):
        pass

    @overload(EdgeIterator, return_type=EdgeDescriptor)
    def edgeIterUnpack(ei):
        return __incidence_and_vertex_list_graph.edgeIterUnpack(ei)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def src(e, g):
        return __incidence_and_vertex_list_graph.src(e, g)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def tgt(e, g):
        return __incidence_and_vertex_list_graph.tgt(e, g)

    @overload(VertexDescriptor, VertexDescriptor, Graph, return_type=EdgeDescriptor)
    def toEdgeDescriptor(v1, v2, g):
        return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g)

    Edge = __incidence_and_vertex_list_graph.Edge
    @overload(Vertex, Vertex, return_type=Edge)
    def makeEdge(s, t):
        return __incidence_and_vertex_list_graph.makeEdge(s, t)

    ColorPropertyMap = __two_bit_color_map.ColorPropertyMap
    @overload(EdgeIterator, VertexVector, Stack, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopRepeat(itr, s1, s2, s3, ctx1, ctx2):
        return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2)

    @overload(EdgeIterator, VertexVector, Stack, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopStep(edgeItr, x, q, c, g, u):
        e = edgeIterUnpack(edgeItr, return_type=EdgeDescriptor)
        v = tgt(e, g, return_type=VertexDescriptor)
        defaultAction(e, g, q, x, return_type=None)
        vc = get(c, v, return_type=Color)
        if (vc) == (white(return_type=Color)):
            if True:
                defaultAction(e, g, q, x, return_type=None)
                put(c, v, gray(return_type=Color), return_type=None)
                defaultAction(v, g, q, x, return_type=None)
                push(v, q, return_type=None)
        else:
            if (vc) == (gray(return_type=Color)):
                if True:
                    defaultAction(e, g, q, x, return_type=None)
            else:
                if True:
                    defaultAction(e, g, q, x, return_type=None)

    __for_iterator_loop3_2 = for_iterator_loop3_2(Graph, VertexDescriptor, EdgeIterator, VertexVector, Stack, ColorPropertyMap, edgeIterEnd, edgeIterNext, bfsInnerLoopStep)
    @overload(VertexVector, Stack, ColorPropertyMap, Graph, return_type=bool)
    def bfsOuterLoopCond(a, q, c, g):
        return not isEmptyStack(q, return_type=bool)

    @overload(VertexVector, Stack, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopRepeat(s1, s2, s3, ctx):
        return __while_loop3.repeat(s1, s2, s3, ctx)

    @overload(VertexVector, Stack, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopStep(x, q, c, g):
        u = top(q, return_type=VertexDescriptor)
        pop(q, return_type=None)
        examineVertex(u, g, q, x, return_type=None)
        edgeItr = EdgeIterator()
        outEdges(u, g, edgeItr, return_type=None)
        bfsInnerLoopRepeat(edgeItr, x, q, c, g, u, return_type=None)
        put(c, u, black(return_type=Color), return_type=None)
        defaultAction(u, g, q, x, return_type=None)

    __while_loop3 = while_loop3(Graph, VertexVector, Stack, ColorPropertyMap, bfsOuterLoopCond, bfsOuterLoopStep)
    @overload(Graph, VertexDescriptor, VertexVector, Stack, ColorPropertyMap, return_type=None)
    def breadthFirstVisit(g, s, a, q, c):
        defaultAction(s, g, q, a, return_type=None)
        push(s, q, return_type=None)
        put(c, s, gray(return_type=Color), return_type=None)
        bfsOuterLoopRepeat(a, q, c, g, return_type=None)

    Color = __two_bit_color_map.Color
    @overload(return_type=Color)
    def black():
        return __two_bit_color_map.black()

    @overload(ColorPropertyMap, VertexDescriptor, return_type=Color)
    def get(pm, k):
        return __two_bit_color_map.get(pm, k)

    @overload(return_type=Color)
    def gray():
        return __two_bit_color_map.gray()

    @overload(VertexIterator, Color, return_type=ColorPropertyMap)
    def initMap(kli, v):
        return __two_bit_color_map.initMap(kli, v)

    @overload(ColorPropertyMap, VertexDescriptor, Color, return_type=None)
    def put(pm, k, v):
        return __two_bit_color_map.put(pm, k, v)

    @overload(return_type=Color)
    def white():
        return __two_bit_color_map.white()

    __namedtuple = namedtuple("PyDFSTestVisitor", ["Color", "ColorPropertyMap", "Edge", "EdgeDescriptor", "EdgeIterator", "EdgeIteratorRange", "Graph", "Int", "Stack", "Vertex", "VertexCount", "VertexDescriptor", "VertexIterator", "VertexVector", "bfsInnerLoopRepeat", "bfsInnerLoopStep", "bfsOuterLoopCond", "bfsOuterLoopRepeat", "bfsOuterLoopStep", "black", "breadthFirstVisit", "defaultAction", "depthFirstSearch", "edgeIterEnd", "edgeIterNext", "edgeIterUnpack", "empty", "emptyStackIsEmpty", "emptyVertexVector", "examineVertex", "get", "gray", "initMap", "isEmptyStack", "iterRangeBegin", "iterRangeEnd", "makeEdge", "makeEdgeIteratorRange", "numVertices", "outDegree", "outEdges", "pop", "push", "pushBack", "pushPopTopBehavior", "put", "src", "tgt", "toEdgeDescriptor", "toVertexDescriptor", "top", "vertexIterEnd", "vertexIterNext", "vertexIterUnpack", "vertices", "white"])
    return __namedtuple(Color, ColorPropertyMap, Edge, EdgeDescriptor, EdgeIterator, EdgeIteratorRange, Graph, Int, Stack, Vertex, VertexCount, VertexDescriptor, VertexIterator, VertexVector, bfsInnerLoopRepeat, bfsInnerLoopStep, bfsOuterLoopCond, bfsOuterLoopRepeat, bfsOuterLoopStep, black, breadthFirstVisit, defaultAction, depthFirstSearch, edgeIterEnd, edgeIterNext, edgeIterUnpack, empty, emptyStackIsEmpty, emptyVertexVector, examineVertex, get, gray, initMap, isEmptyStack, iterRangeBegin, iterRangeEnd, makeEdge, makeEdgeIteratorRange, numVertices, outDegree, outEdges, pop, push, pushBack, pushPopTopBehavior, put, src, tgt, toEdgeDescriptor, toVertexDescriptor, top, vertexIterEnd, vertexIterNext, vertexIterUnpack, vertices, white)


def PyDijkstraVisitor():
    overload = functools.partial(multiple_dispatch.overload, {})
    __base_types = base_types()
    __base_float_ops = base_float_ops()
    Vertex = __base_types.Vertex
    __incidence_and_vertex_list_graph = incidence_and_vertex_list_graph(Vertex)
    VertexCount = __incidence_and_vertex_list_graph.VertexCount
    VertexDescriptor = __incidence_and_vertex_list_graph.VertexDescriptor
    __vector = vector(VertexDescriptor)
    VertexVector = __vector.Vector
    @overload(return_type=VertexVector)
    def emptyVertexVector():
        return __vector.empty()

    @overload(VertexDescriptor, VertexVector, return_type=None)
    def pushBack(a, v):
        return __vector.pushBack(a, v)

    VertexIterator = __incidence_and_vertex_list_graph.VertexIterator
    @overload(VertexIterator, return_type=bool)
    def vertexIterEnd(ei):
        return __incidence_and_vertex_list_graph.vertexIterEnd(ei)

    @overload(VertexIterator, return_type=None)
    def vertexIterNext(ei):
        return __incidence_and_vertex_list_graph.vertexIterNext(ei)

    @overload(VertexIterator, return_type=VertexDescriptor)
    def vertexIterUnpack(ei):
        return __incidence_and_vertex_list_graph.vertexIterUnpack(ei)

    __read_write_property_map1 = read_write_property_map(VertexDescriptor, VertexIterator, VertexDescriptor, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    VertexPredecessorMap = __read_write_property_map1.PropertyMap
    @overload(return_type=VertexPredecessorMap)
    def emptyVPMap():
        return __read_write_property_map1.emptyMap()

    @overload(VertexIterator, VertexPredecessorMap, VertexDescriptor, return_type=None)
    def forIterationEnd(itr, state, ctx):
        mut_state = state
        if vertexIterEnd(itr, return_type=bool):
            if True:
                populateVPMapLoopRepeat(itr, mut_state, ctx, return_type=None)
                assert (mut_state) == (state)
        else:
            pass

    @overload(VertexPredecessorMap, VertexDescriptor, return_type=VertexDescriptor)
    def get(pm, k):
        return __read_write_property_map1.get(pm, k)

    @overload(VertexIterator, VertexDescriptor, return_type=VertexPredecessorMap)
    def initMap(kli, v):
        return __read_write_property_map1.initMap(kli, v)

    @overload(VertexIterator, VertexPredecessorMap, VertexDescriptor, return_type=None)
    def populateVPMapLoopRepeat(itr, state, ctx):
        return __for_iterator_loop.forLoopRepeat(itr, state, ctx)

    @overload(VertexIterator, VertexPredecessorMap, VertexDescriptor, return_type=None)
    def populateVPMapLoopStep(itr, vpm, vd):
        v = vertexIterUnpack(itr, return_type=VertexDescriptor)
        put(vpm, v, v, return_type=None)

    __for_iterator_loop = for_iterator_loop(VertexDescriptor, VertexIterator, VertexPredecessorMap, vertexIterEnd, vertexIterNext, populateVPMapLoopStep)
    @overload(VertexPredecessorMap, VertexDescriptor, VertexDescriptor, return_type=None)
    def put(pm, k, v):
        return __read_write_property_map1.put(pm, k, v)

    __two_bit_color_map = two_bit_color_map(VertexDescriptor, VertexIterator, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    Int = __base_types.Int
    Graph = __incidence_and_vertex_list_graph.Graph
    @overload(Graph, return_type=VertexCount)
    def numVertices(g):
        return __incidence_and_vertex_list_graph.numVertices(g)

    @overload(VertexDescriptor, Graph, return_type=VertexCount)
    def outDegree(v, g):
        return __incidence_and_vertex_list_graph.outDegree(v, g)

    @overload(Vertex, Graph, return_type=VertexDescriptor)
    def toVertexDescriptor(v, g):
        return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g)

    @overload(Graph, VertexIterator, return_type=None)
    def vertices(g, itr):
        return __incidence_and_vertex_list_graph.vertices(g, itr)

    EdgeIterator = __incidence_and_vertex_list_graph.EdgeIterator
    __pair = pair(EdgeIterator, EdgeIterator)
    EdgeIteratorRange = __pair.Pair
    @overload(EdgeIterator, return_type=bool)
    def edgeIterEnd(ei):
        return __incidence_and_vertex_list_graph.edgeIterEnd(ei)

    @overload(EdgeIterator, return_type=None)
    def edgeIterNext(ei):
        return __incidence_and_vertex_list_graph.edgeIterNext(ei)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeBegin(p):
        return __pair.first(p)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeEnd(p):
        return __pair.second(p)

    @overload(EdgeIterator, EdgeIterator, return_type=EdgeIteratorRange)
    def makeEdgeIteratorRange(a, b):
        return __pair.makePair(a, b)

    @overload(VertexDescriptor, Graph, EdgeIterator, return_type=None)
    def outEdges(v, g, itr):
        return __incidence_and_vertex_list_graph.outEdges(v, g, itr)

    EdgeDescriptor = __incidence_and_vertex_list_graph.EdgeDescriptor
    @overload(EdgeIterator, return_type=EdgeDescriptor)
    def edgeIterUnpack(ei):
        return __incidence_and_vertex_list_graph.edgeIterUnpack(ei)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def src(e, g):
        return __incidence_and_vertex_list_graph.src(e, g)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def tgt(e, g):
        return __incidence_and_vertex_list_graph.tgt(e, g)

    @overload(VertexDescriptor, VertexDescriptor, Graph, return_type=EdgeDescriptor)
    def toEdgeDescriptor(v1, v2, g):
        return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g)

    Edge = __incidence_and_vertex_list_graph.Edge
    @overload(Vertex, Vertex, return_type=Edge)
    def makeEdge(s, t):
        return __incidence_and_vertex_list_graph.makeEdge(s, t)

    Cost = __base_float_ops.Float
    __read_write_property_map = read_write_property_map(EdgeDescriptor, EdgeIterator, Cost, edgeIterEnd, edgeIterNext, edgeIterUnpack)
    EdgeCostMap = __read_write_property_map.PropertyMap
    @overload(return_type=EdgeCostMap)
    def emptyECMap():
        return __read_write_property_map.emptyMap()

    __read_write_property_map0 = read_write_property_map(VertexDescriptor, VertexIterator, Cost, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    VertexCostMap = __read_write_property_map0.PropertyMap
    __priority_queue = priority_queue(VertexDescriptor, Cost, VertexCostMap, get)
    PriorityQueue = __priority_queue.PriorityQueue
    @overload(PriorityQueue, return_type=VertexDescriptor)
    def front(q):
        return __priority_queue.front(q)

    @overload(PriorityQueue, return_type=bool)
    def isEmptyQueue(q):
        return __priority_queue.isEmpty(q)

    @overload(PriorityQueue, return_type=None)
    def pop(q):
        return __priority_queue.pop(q)

    @overload(VertexDescriptor, PriorityQueue, return_type=None)
    def push(a, q):
        return __priority_queue.push(a, q)

    __triplet = triplet(VertexCostMap, VertexPredecessorMap, EdgeCostMap)
    StateWithMaps = __triplet.Triplet
    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def blackTarget(edgeOrVertex, g, q, a):
        pass

    @overload(VertexDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def discoverVertex(edgeOrVertex, g, q, a):
        pass

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def examineEdge(e, g, pq, swm):
        origVcm = getVertexCostMap(swm, return_type=VertexCostMap)
        vpm = getVertexPredecessorMap(swm, return_type=VertexPredecessorMap)
        ecm = getEdgeCostMap(swm, return_type=EdgeCostMap)
        vcm = origVcm
        relax(e, g, ecm, vcm, vpm, return_type=None)
        if (vcm) == (origVcm):
            pass
        else:
            if True:
                swm.mutate(putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm, return_type=StateWithMaps), return_type=StateWithMaps))
                pq.mutate(update(vcm, tgt(e, g, return_type=VertexDescriptor), pq, return_type=PriorityQueue))

    @overload(VertexDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def examineVertex(edgeOrVertex, g, q, a):
        pass

    @overload(VertexDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def finishVertex(edgeOrVertex, g, q, a):
        pass

    @overload(StateWithMaps, return_type=EdgeCostMap)
    def getEdgeCostMap(p):
        return __triplet.third(p)

    @overload(StateWithMaps, return_type=VertexPredecessorMap)
    def getVertexPredecessorMap(p):
        return __triplet.second(p)

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def grayTarget(edgeOrVertex, g, q, a):
        pass

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def nonTreeEdge(edgeOrVertex, g, q, a):
        pass

    @overload(VertexPredecessorMap, StateWithMaps, return_type=StateWithMaps)
    def putVertexPredecessorMap(vpm, swm):
        return makeStateWithMaps(getVertexCostMap(swm, return_type=VertexCostMap), vpm, getEdgeCostMap(swm, return_type=EdgeCostMap), return_type=StateWithMaps)

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def treeEdge(edgeOrVertex, g, q, a):
        pass

    @overload(VertexCostMap, return_type=PriorityQueue)
    def emptyPriorityQueue(pm):
        return __priority_queue.empty(pm)

    @overload(return_type=VertexCostMap)
    def emptyVCMap():
        return __read_write_property_map0.emptyMap()

    @overload(StateWithMaps, return_type=VertexCostMap)
    def getVertexCostMap(p):
        return __triplet.first(p)

    @overload(VertexCostMap, VertexPredecessorMap, EdgeCostMap, return_type=StateWithMaps)
    def makeStateWithMaps(a, b, c):
        return __triplet.makeTriplet(a, b, c)

    @overload(VertexCostMap, StateWithMaps, return_type=StateWithMaps)
    def putVertexCostMap(vcm, swm):
        return makeStateWithMaps(vcm, getVertexPredecessorMap(swm, return_type=VertexPredecessorMap), getEdgeCostMap(swm, return_type=EdgeCostMap), return_type=StateWithMaps)

    @overload(EdgeDescriptor, Graph, EdgeCostMap, VertexCostMap, VertexPredecessorMap, return_type=None)
    def relax(e, g, ecm, vcm, vpm):
        u = src(e, g, return_type=VertexDescriptor)
        v = tgt(e, g, return_type=VertexDescriptor)
        uCost = get(vcm, u, return_type=Cost)
        vCost = get(vcm, v, return_type=Cost)
        edgeCost = get(ecm, e, return_type=Cost)
        if less(plus(uCost, edgeCost, return_type=Cost), vCost, return_type=bool):
            if True:
                put(vcm, v, plus(uCost, edgeCost, return_type=Cost), return_type=None)
                put(vpm, v, u, return_type=None)
        else:
            pass

    @overload(VertexCostMap, VertexDescriptor, PriorityQueue, return_type=PriorityQueue)
    def update(pm, a, pq):
        return __priority_queue.update(pm, a, pq)

    @overload(Graph, VertexDescriptor, VertexCostMap, EdgeCostMap, Cost, VertexPredecessorMap, return_type=None)
    def dijkstraShortestPaths(g, start, vcm, ecm, initialCost, vpm):
        put(vcm, start, initialCost, return_type=None)
        vertexItr = VertexIterator()
        vertices(g, vertexItr, return_type=None)
        vpm.mutate(emptyVPMap(return_type=VertexPredecessorMap))
        populateVPMapLoopRepeat(vertexItr, vpm, start, return_type=None)
        pq = emptyPriorityQueue(vcm, return_type=PriorityQueue)
        swm = makeStateWithMaps(vcm, vpm, ecm, return_type=StateWithMaps)
        c = initMap(vertexItr, white(return_type=Color), return_type=ColorPropertyMap)
        breadthFirstVisit(g, start, swm, pq, c, return_type=None)
        vcm.mutate(getVertexCostMap(swm, return_type=VertexCostMap))
        vpm.mutate(getVertexPredecessorMap(swm, return_type=VertexPredecessorMap))

    @overload(VertexCostMap, VertexDescriptor, return_type=Cost)
    def get(pm, k):
        return __read_write_property_map0.get(pm, k)

    @overload(EdgeCostMap, EdgeDescriptor, return_type=Cost)
    def get(pm, k):
        return __read_write_property_map.get(pm, k)

    @overload(VertexIterator, Cost, return_type=VertexCostMap)
    def initMap(kli, v):
        return __read_write_property_map0.initMap(kli, v)

    @overload(EdgeIterator, Cost, return_type=EdgeCostMap)
    def initMap(kli, v):
        return __read_write_property_map.initMap(kli, v)

    @overload(Cost, Cost, return_type=bool)
    def less(i1, i2):
        return __base_float_ops.less(i1, i2)

    @overload(Cost, Cost, return_type=Cost)
    def plus(i1, i2):
        return __base_float_ops.plus(i1, i2)

    @overload(VertexCostMap, VertexDescriptor, Cost, return_type=None)
    def put(pm, k, v):
        return __read_write_property_map0.put(pm, k, v)

    @overload(EdgeCostMap, EdgeDescriptor, Cost, return_type=None)
    def put(pm, k, v):
        return __read_write_property_map.put(pm, k, v)

    ColorPropertyMap = __two_bit_color_map.ColorPropertyMap
    @overload(EdgeIterator, StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopRepeat(itr, s1, s2, s3, ctx1, ctx2):
        return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2)

    @overload(EdgeIterator, StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopStep(edgeItr, x, q, c, g, u):
        e = edgeIterUnpack(edgeItr, return_type=EdgeDescriptor)
        v = tgt(e, g, return_type=VertexDescriptor)
        examineEdge(e, g, q, x, return_type=None)
        vc = get(c, v, return_type=Color)
        if (vc) == (white(return_type=Color)):
            if True:
                treeEdge(e, g, q, x, return_type=None)
                put(c, v, gray(return_type=Color), return_type=None)
                discoverVertex(v, g, q, x, return_type=None)
                push(v, q, return_type=None)
        else:
            if (vc) == (gray(return_type=Color)):
                if True:
                    grayTarget(e, g, q, x, return_type=None)
            else:
                if True:
                    blackTarget(e, g, q, x, return_type=None)

    __for_iterator_loop3_2 = for_iterator_loop3_2(Graph, VertexDescriptor, EdgeIterator, StateWithMaps, PriorityQueue, ColorPropertyMap, edgeIterEnd, edgeIterNext, bfsInnerLoopStep)
    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, return_type=bool)
    def bfsOuterLoopCond(a, q, c, g):
        return not isEmptyQueue(q, return_type=bool)

    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopRepeat(s1, s2, s3, ctx):
        return __while_loop3.repeat(s1, s2, s3, ctx)

    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopStep(x, q, c, g):
        u = front(q, return_type=VertexDescriptor)
        pop(q, return_type=None)
        examineVertex(u, g, q, x, return_type=None)
        edgeItr = EdgeIterator()
        outEdges(u, g, edgeItr, return_type=None)
        bfsInnerLoopRepeat(edgeItr, x, q, c, g, u, return_type=None)
        put(c, u, black(return_type=Color), return_type=None)
        finishVertex(u, g, q, x, return_type=None)

    __while_loop3 = while_loop3(Graph, StateWithMaps, PriorityQueue, ColorPropertyMap, bfsOuterLoopCond, bfsOuterLoopStep)
    @overload(Graph, VertexDescriptor, StateWithMaps, PriorityQueue, ColorPropertyMap, return_type=None)
    def breadthFirstVisit(g, s, a, q, c):
        discoverVertex(s, g, q, a, return_type=None)
        push(s, q, return_type=None)
        put(c, s, gray(return_type=Color), return_type=None)
        bfsOuterLoopRepeat(a, q, c, g, return_type=None)

    Color = __two_bit_color_map.Color
    @overload(return_type=Color)
    def black():
        return __two_bit_color_map.black()

    @overload(ColorPropertyMap, VertexDescriptor, return_type=Color)
    def get(pm, k):
        return __two_bit_color_map.get(pm, k)

    @overload(return_type=Color)
    def gray():
        return __two_bit_color_map.gray()

    @overload(VertexIterator, Color, return_type=ColorPropertyMap)
    def initMap(kli, v):
        return __two_bit_color_map.initMap(kli, v)

    @overload(ColorPropertyMap, VertexDescriptor, Color, return_type=None)
    def put(pm, k, v):
        return __two_bit_color_map.put(pm, k, v)

    @overload(return_type=Color)
    def white():
        return __two_bit_color_map.white()

    __namedtuple = namedtuple("PyDijkstraVisitor", ["Color", "ColorPropertyMap", "Cost", "Edge", "EdgeCostMap", "EdgeDescriptor", "EdgeIterator", "EdgeIteratorRange", "Graph", "Int", "PriorityQueue", "StateWithMaps", "Vertex", "VertexCostMap", "VertexCount", "VertexDescriptor", "VertexIterator", "VertexPredecessorMap", "VertexVector", "bfsInnerLoopRepeat", "bfsInnerLoopStep", "bfsOuterLoopCond", "bfsOuterLoopRepeat", "bfsOuterLoopStep", "black", "blackTarget", "breadthFirstVisit", "dijkstraShortestPaths", "discoverVertex", "edgeIterEnd", "edgeIterNext", "edgeIterUnpack", "emptyECMap", "emptyPriorityQueue", "emptyVCMap", "emptyVPMap", "emptyVertexVector", "examineEdge", "examineVertex", "finishVertex", "forIterationEnd", "front", "get", "getEdgeCostMap", "getVertexCostMap", "getVertexPredecessorMap", "gray", "grayTarget", "initMap", "isEmptyQueue", "iterRangeBegin", "iterRangeEnd", "less", "makeEdge", "makeEdgeIteratorRange", "makeStateWithMaps", "nonTreeEdge", "numVertices", "outDegree", "outEdges", "plus", "pop", "populateVPMapLoopRepeat", "populateVPMapLoopStep", "push", "pushBack", "put", "putVertexCostMap", "putVertexPredecessorMap", "relax", "src", "tgt", "toEdgeDescriptor", "toVertexDescriptor", "treeEdge", "update", "vertexIterEnd", "vertexIterNext", "vertexIterUnpack", "vertices", "white"])
    return __namedtuple(Color, ColorPropertyMap, Cost, Edge, EdgeCostMap, EdgeDescriptor, EdgeIterator, EdgeIteratorRange, Graph, Int, PriorityQueue, StateWithMaps, Vertex, VertexCostMap, VertexCount, VertexDescriptor, VertexIterator, VertexPredecessorMap, VertexVector, bfsInnerLoopRepeat, bfsInnerLoopStep, bfsOuterLoopCond, bfsOuterLoopRepeat, bfsOuterLoopStep, black, blackTarget, breadthFirstVisit, dijkstraShortestPaths, discoverVertex, edgeIterEnd, edgeIterNext, edgeIterUnpack, emptyECMap, emptyPriorityQueue, emptyVCMap, emptyVPMap, emptyVertexVector, examineEdge, examineVertex, finishVertex, forIterationEnd, front, get, getEdgeCostMap, getVertexCostMap, getVertexPredecessorMap, gray, grayTarget, initMap, isEmptyQueue, iterRangeBegin, iterRangeEnd, less, makeEdge, makeEdgeIteratorRange, makeStateWithMaps, nonTreeEdge, numVertices, outDegree, outEdges, plus, pop, populateVPMapLoopRepeat, populateVPMapLoopStep, push, pushBack, put, putVertexCostMap, putVertexPredecessorMap, relax, src, tgt, toEdgeDescriptor, toVertexDescriptor, treeEdge, update, vertexIterEnd, vertexIterNext, vertexIterUnpack, vertices, white)


def PyPrimVisitor():
    overload = functools.partial(multiple_dispatch.overload, {})
    __base_types = base_types()
    __base_float_ops = base_float_ops()
    Vertex = __base_types.Vertex
    __incidence_and_vertex_list_graph = incidence_and_vertex_list_graph(Vertex)
    VertexCount = __incidence_and_vertex_list_graph.VertexCount
    VertexDescriptor = __incidence_and_vertex_list_graph.VertexDescriptor
    __vector = vector(VertexDescriptor)
    VertexVector = __vector.Vector
    @overload(return_type=VertexVector)
    def emptyVertexVector():
        return __vector.empty()

    @overload(VertexDescriptor, VertexVector, return_type=None)
    def pushBack(a, v):
        return __vector.pushBack(a, v)

    VertexIterator = __incidence_and_vertex_list_graph.VertexIterator
    @overload(VertexIterator, return_type=bool)
    def vertexIterEnd(ei):
        return __incidence_and_vertex_list_graph.vertexIterEnd(ei)

    @overload(VertexIterator, return_type=None)
    def vertexIterNext(ei):
        return __incidence_and_vertex_list_graph.vertexIterNext(ei)

    @overload(VertexIterator, return_type=VertexDescriptor)
    def vertexIterUnpack(ei):
        return __incidence_and_vertex_list_graph.vertexIterUnpack(ei)

    __read_write_property_map1 = read_write_property_map(VertexDescriptor, VertexIterator, VertexDescriptor, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    VertexPredecessorMap = __read_write_property_map1.PropertyMap
    @overload(return_type=VertexPredecessorMap)
    def emptyVPMap():
        return __read_write_property_map1.emptyMap()

    @overload(VertexIterator, VertexPredecessorMap, VertexDescriptor, return_type=None)
    def forIterationEnd(itr, state, ctx):
        mut_state = state
        if vertexIterEnd(itr, return_type=bool):
            if True:
                populateVPMapLoopRepeat(itr, mut_state, ctx, return_type=None)
                assert (mut_state) == (state)
        else:
            pass

    @overload(VertexPredecessorMap, VertexDescriptor, return_type=VertexDescriptor)
    def get(pm, k):
        return __read_write_property_map1.get(pm, k)

    @overload(VertexIterator, VertexDescriptor, return_type=VertexPredecessorMap)
    def initMap(kli, v):
        return __read_write_property_map1.initMap(kli, v)

    @overload(VertexIterator, VertexPredecessorMap, VertexDescriptor, return_type=None)
    def populateVPMapLoopRepeat(itr, state, ctx):
        return __for_iterator_loop.forLoopRepeat(itr, state, ctx)

    @overload(VertexIterator, VertexPredecessorMap, VertexDescriptor, return_type=None)
    def populateVPMapLoopStep(itr, vpm, vd):
        v = vertexIterUnpack(itr, return_type=VertexDescriptor)
        put(vpm, v, v, return_type=None)

    __for_iterator_loop = for_iterator_loop(VertexDescriptor, VertexIterator, VertexPredecessorMap, vertexIterEnd, vertexIterNext, populateVPMapLoopStep)
    @overload(VertexPredecessorMap, VertexDescriptor, VertexDescriptor, return_type=None)
    def put(pm, k, v):
        return __read_write_property_map1.put(pm, k, v)

    __two_bit_color_map = two_bit_color_map(VertexDescriptor, VertexIterator, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    Int = __base_types.Int
    Graph = __incidence_and_vertex_list_graph.Graph
    @overload(Graph, return_type=VertexCount)
    def numVertices(g):
        return __incidence_and_vertex_list_graph.numVertices(g)

    @overload(VertexDescriptor, Graph, return_type=VertexCount)
    def outDegree(v, g):
        return __incidence_and_vertex_list_graph.outDegree(v, g)

    @overload(Vertex, Graph, return_type=VertexDescriptor)
    def toVertexDescriptor(v, g):
        return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g)

    @overload(Graph, VertexIterator, return_type=None)
    def vertices(g, itr):
        return __incidence_and_vertex_list_graph.vertices(g, itr)

    EdgeIterator = __incidence_and_vertex_list_graph.EdgeIterator
    __pair = pair(EdgeIterator, EdgeIterator)
    EdgeIteratorRange = __pair.Pair
    @overload(EdgeIterator, return_type=bool)
    def edgeIterEnd(ei):
        return __incidence_and_vertex_list_graph.edgeIterEnd(ei)

    @overload(EdgeIterator, return_type=None)
    def edgeIterNext(ei):
        return __incidence_and_vertex_list_graph.edgeIterNext(ei)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeBegin(p):
        return __pair.first(p)

    @overload(EdgeIteratorRange, return_type=EdgeIterator)
    def iterRangeEnd(p):
        return __pair.second(p)

    @overload(EdgeIterator, EdgeIterator, return_type=EdgeIteratorRange)
    def makeEdgeIteratorRange(a, b):
        return __pair.makePair(a, b)

    @overload(VertexDescriptor, Graph, EdgeIterator, return_type=None)
    def outEdges(v, g, itr):
        return __incidence_and_vertex_list_graph.outEdges(v, g, itr)

    EdgeDescriptor = __incidence_and_vertex_list_graph.EdgeDescriptor
    @overload(EdgeIterator, return_type=EdgeDescriptor)
    def edgeIterUnpack(ei):
        return __incidence_and_vertex_list_graph.edgeIterUnpack(ei)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def src(e, g):
        return __incidence_and_vertex_list_graph.src(e, g)

    @overload(EdgeDescriptor, Graph, return_type=VertexDescriptor)
    def tgt(e, g):
        return __incidence_and_vertex_list_graph.tgt(e, g)

    @overload(VertexDescriptor, VertexDescriptor, Graph, return_type=EdgeDescriptor)
    def toEdgeDescriptor(v1, v2, g):
        return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g)

    Edge = __incidence_and_vertex_list_graph.Edge
    @overload(Vertex, Vertex, return_type=Edge)
    def makeEdge(s, t):
        return __incidence_and_vertex_list_graph.makeEdge(s, t)

    Cost = __base_float_ops.Float
    __read_write_property_map = read_write_property_map(EdgeDescriptor, EdgeIterator, Cost, edgeIterEnd, edgeIterNext, edgeIterUnpack)
    EdgeCostMap = __read_write_property_map.PropertyMap
    @overload(return_type=EdgeCostMap)
    def emptyECMap():
        return __read_write_property_map.emptyMap()

    __read_write_property_map0 = read_write_property_map(VertexDescriptor, VertexIterator, Cost, vertexIterEnd, vertexIterNext, vertexIterUnpack)
    VertexCostMap = __read_write_property_map0.PropertyMap
    __triplet = triplet(VertexCostMap, VertexPredecessorMap, EdgeCostMap)
    StateWithMaps = __triplet.Triplet
    @overload(StateWithMaps, return_type=EdgeCostMap)
    def getEdgeCostMap(p):
        return __triplet.third(p)

    @overload(StateWithMaps, return_type=VertexPredecessorMap)
    def getVertexPredecessorMap(p):
        return __triplet.second(p)

    @overload(VertexPredecessorMap, StateWithMaps, return_type=StateWithMaps)
    def putVertexPredecessorMap(vpm, swm):
        return makeStateWithMaps(getVertexCostMap(swm, return_type=VertexCostMap), vpm, getEdgeCostMap(swm, return_type=EdgeCostMap), return_type=StateWithMaps)

    @overload(return_type=VertexCostMap)
    def emptyVCMap():
        return __read_write_property_map0.emptyMap()

    @overload(StateWithMaps, return_type=VertexCostMap)
    def getVertexCostMap(p):
        return __triplet.first(p)

    @overload(VertexCostMap, VertexPredecessorMap, EdgeCostMap, return_type=StateWithMaps)
    def makeStateWithMaps(a, b, c):
        return __triplet.makeTriplet(a, b, c)

    @overload(VertexCostMap, StateWithMaps, return_type=StateWithMaps)
    def putVertexCostMap(vcm, swm):
        return makeStateWithMaps(vcm, getVertexPredecessorMap(swm, return_type=VertexPredecessorMap), getEdgeCostMap(swm, return_type=EdgeCostMap), return_type=StateWithMaps)

    @overload(EdgeDescriptor, Graph, EdgeCostMap, VertexCostMap, VertexPredecessorMap, return_type=None)
    def relax(e, g, ecm, vcm, vpm):
        u = src(e, g, return_type=VertexDescriptor)
        v = tgt(e, g, return_type=VertexDescriptor)
        uCost = get(vcm, u, return_type=Cost)
        vCost = get(vcm, v, return_type=Cost)
        edgeCost = get(ecm, e, return_type=Cost)
        if less(second(uCost, edgeCost, return_type=Cost), vCost, return_type=bool):
            if True:
                put(vcm, v, second(uCost, edgeCost, return_type=Cost), return_type=None)
                put(vpm, v, u, return_type=None)
        else:
            pass

    @overload(VertexCostMap, VertexDescriptor, return_type=Cost)
    def get(pm, k):
        return __read_write_property_map0.get(pm, k)

    @overload(EdgeCostMap, EdgeDescriptor, return_type=Cost)
    def get(pm, k):
        return __read_write_property_map.get(pm, k)

    @overload(VertexIterator, Cost, return_type=VertexCostMap)
    def initMap(kli, v):
        return __read_write_property_map0.initMap(kli, v)

    @overload(EdgeIterator, Cost, return_type=EdgeCostMap)
    def initMap(kli, v):
        return __read_write_property_map.initMap(kli, v)

    @overload(Cost, Cost, return_type=bool)
    def less(i1, i2):
        return __base_float_ops.less(i1, i2)

    @overload(Cost, Cost, return_type=Cost)
    def plus(i1, i2):
        return __base_float_ops.plus(i1, i2)

    @overload(Graph, VertexDescriptor, VertexCostMap, EdgeCostMap, Cost, VertexPredecessorMap, return_type=None)
    def primMinimumSpanningTree(g, start, vcm, ecm, initialCost, vpm):
        put(vcm, start, initialCost, return_type=None)
        vertexItr = VertexIterator()
        vertices(g, vertexItr, return_type=None)
        vpm.mutate(emptyVPMap(return_type=VertexPredecessorMap))
        populateVPMapLoopRepeat(vertexItr, vpm, start, return_type=None)
        pq = emptyPriorityQueue(vcm, return_type=PriorityQueue)
        swm = makeStateWithMaps(vcm, vpm, ecm, return_type=StateWithMaps)
        c = initMap(vertexItr, white(return_type=Color), return_type=ColorPropertyMap)
        breadthFirstVisit(g, start, swm, pq, c, return_type=None)
        vcm.mutate(getVertexCostMap(swm, return_type=VertexCostMap))
        vpm.mutate(getVertexPredecessorMap(swm, return_type=VertexPredecessorMap))

    @overload(VertexCostMap, VertexDescriptor, Cost, return_type=None)
    def put(pm, k, v):
        return __read_write_property_map0.put(pm, k, v)

    @overload(EdgeCostMap, EdgeDescriptor, Cost, return_type=None)
    def put(pm, k, v):
        return __read_write_property_map.put(pm, k, v)

    @overload(Cost, Cost, return_type=Cost)
    def second(c1, c2):
        return c2

    ColorPropertyMap = __two_bit_color_map.ColorPropertyMap
    Color = __two_bit_color_map.Color
    @overload(return_type=Color)
    def black():
        return __two_bit_color_map.black()

    @overload(ColorPropertyMap, VertexDescriptor, return_type=Color)
    def get(pm, k):
        return __two_bit_color_map.get(pm, k)

    __priority_queue = priority_queue(VertexDescriptor, Cost, VertexCostMap, get)
    PriorityQueue = __priority_queue.PriorityQueue
    @overload(EdgeIterator, StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopRepeat(itr, s1, s2, s3, ctx1, ctx2):
        return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2)

    @overload(EdgeIterator, StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, VertexDescriptor, return_type=None)
    def bfsInnerLoopStep(edgeItr, x, q, c, g, u):
        e = edgeIterUnpack(edgeItr, return_type=EdgeDescriptor)
        v = tgt(e, g, return_type=VertexDescriptor)
        examineEdge(e, g, q, x, return_type=None)
        vc = get(c, v, return_type=Color)
        if (vc) == (white(return_type=Color)):
            if True:
                treeEdge(e, g, q, x, return_type=None)
                put(c, v, gray(return_type=Color), return_type=None)
                discoverVertex(v, g, q, x, return_type=None)
                push(v, q, return_type=None)
        else:
            if (vc) == (gray(return_type=Color)):
                if True:
                    grayTarget(e, g, q, x, return_type=None)
            else:
                if True:
                    blackTarget(e, g, q, x, return_type=None)

    __for_iterator_loop3_2 = for_iterator_loop3_2(Graph, VertexDescriptor, EdgeIterator, StateWithMaps, PriorityQueue, ColorPropertyMap, edgeIterEnd, edgeIterNext, bfsInnerLoopStep)
    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, return_type=bool)
    def bfsOuterLoopCond(a, q, c, g):
        return not isEmptyQueue(q, return_type=bool)

    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopRepeat(s1, s2, s3, ctx):
        return __while_loop3.repeat(s1, s2, s3, ctx)

    @overload(StateWithMaps, PriorityQueue, ColorPropertyMap, Graph, return_type=None)
    def bfsOuterLoopStep(x, q, c, g):
        u = front(q, return_type=VertexDescriptor)
        pop(q, return_type=None)
        examineVertex(u, g, q, x, return_type=None)
        edgeItr = EdgeIterator()
        outEdges(u, g, edgeItr, return_type=None)
        bfsInnerLoopRepeat(edgeItr, x, q, c, g, u, return_type=None)
        put(c, u, black(return_type=Color), return_type=None)
        finishVertex(u, g, q, x, return_type=None)

    __while_loop3 = while_loop3(Graph, StateWithMaps, PriorityQueue, ColorPropertyMap, bfsOuterLoopCond, bfsOuterLoopStep)
    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def blackTarget(edgeOrVertex, g, q, a):
        pass

    @overload(Graph, VertexDescriptor, StateWithMaps, PriorityQueue, ColorPropertyMap, return_type=None)
    def breadthFirstVisit(g, s, a, q, c):
        discoverVertex(s, g, q, a, return_type=None)
        push(s, q, return_type=None)
        put(c, s, gray(return_type=Color), return_type=None)
        bfsOuterLoopRepeat(a, q, c, g, return_type=None)

    @overload(VertexDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def discoverVertex(edgeOrVertex, g, q, a):
        pass

    @overload(VertexCostMap, return_type=PriorityQueue)
    def emptyPriorityQueue(pm):
        return __priority_queue.empty(pm)

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def examineEdge(e, g, pq, swm):
        origVcm = getVertexCostMap(swm, return_type=VertexCostMap)
        vpm = getVertexPredecessorMap(swm, return_type=VertexPredecessorMap)
        ecm = getEdgeCostMap(swm, return_type=EdgeCostMap)
        vcm = origVcm
        relax(e, g, ecm, vcm, vpm, return_type=None)
        if (vcm) == (origVcm):
            pass
        else:
            if True:
                swm.mutate(putVertexPredecessorMap(vpm, putVertexCostMap(vcm, swm, return_type=StateWithMaps), return_type=StateWithMaps))
                pq.mutate(update(vcm, tgt(e, g, return_type=VertexDescriptor), pq, return_type=PriorityQueue))

    @overload(VertexDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def examineVertex(edgeOrVertex, g, q, a):
        pass

    @overload(VertexDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def finishVertex(edgeOrVertex, g, q, a):
        pass

    @overload(PriorityQueue, return_type=VertexDescriptor)
    def front(q):
        return __priority_queue.front(q)

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def grayTarget(edgeOrVertex, g, q, a):
        pass

    @overload(PriorityQueue, return_type=bool)
    def isEmptyQueue(q):
        return __priority_queue.isEmpty(q)

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def nonTreeEdge(edgeOrVertex, g, q, a):
        pass

    @overload(PriorityQueue, return_type=None)
    def pop(q):
        return __priority_queue.pop(q)

    @overload(VertexDescriptor, PriorityQueue, return_type=None)
    def push(a, q):
        return __priority_queue.push(a, q)

    @overload(EdgeDescriptor, Graph, PriorityQueue, StateWithMaps, return_type=None)
    def treeEdge(edgeOrVertex, g, q, a):
        pass

    @overload(VertexCostMap, VertexDescriptor, PriorityQueue, return_type=PriorityQueue)
    def update(pm, a, pq):
        return __priority_queue.update(pm, a, pq)

    @overload(return_type=Color)
    def gray():
        return __two_bit_color_map.gray()

    @overload(VertexIterator, Color, return_type=ColorPropertyMap)
    def initMap(kli, v):
        return __two_bit_color_map.initMap(kli, v)

    @overload(ColorPropertyMap, VertexDescriptor, Color, return_type=None)
    def put(pm, k, v):
        return __two_bit_color_map.put(pm, k, v)

    @overload(return_type=Color)
    def white():
        return __two_bit_color_map.white()

    __namedtuple = namedtuple("PyPrimVisitor", ["Color", "ColorPropertyMap", "Cost", "Edge", "EdgeCostMap", "EdgeDescriptor", "EdgeIterator", "EdgeIteratorRange", "Graph", "Int", "PriorityQueue", "StateWithMaps", "Vertex", "VertexCostMap", "VertexCount", "VertexDescriptor", "VertexIterator", "VertexPredecessorMap", "VertexVector", "bfsInnerLoopRepeat", "bfsInnerLoopStep", "bfsOuterLoopCond", "bfsOuterLoopRepeat", "bfsOuterLoopStep", "black", "blackTarget", "breadthFirstVisit", "discoverVertex", "edgeIterEnd", "edgeIterNext", "edgeIterUnpack", "emptyECMap", "emptyPriorityQueue", "emptyVCMap", "emptyVPMap", "emptyVertexVector", "examineEdge", "examineVertex", "finishVertex", "forIterationEnd", "front", "get", "getEdgeCostMap", "getVertexCostMap", "getVertexPredecessorMap", "gray", "grayTarget", "initMap", "isEmptyQueue", "iterRangeBegin", "iterRangeEnd", "less", "makeEdge", "makeEdgeIteratorRange", "makeStateWithMaps", "nonTreeEdge", "numVertices", "outDegree", "outEdges", "plus", "pop", "populateVPMapLoopRepeat", "populateVPMapLoopStep", "primMinimumSpanningTree", "push", "pushBack", "put", "putVertexCostMap", "putVertexPredecessorMap", "relax", "second", "src", "tgt", "toEdgeDescriptor", "toVertexDescriptor", "treeEdge", "update", "vertexIterEnd", "vertexIterNext", "vertexIterUnpack", "vertices", "white"])
    return __namedtuple(Color, ColorPropertyMap, Cost, Edge, EdgeCostMap, EdgeDescriptor, EdgeIterator, EdgeIteratorRange, Graph, Int, PriorityQueue, StateWithMaps, Vertex, VertexCostMap, VertexCount, VertexDescriptor, VertexIterator, VertexPredecessorMap, VertexVector, bfsInnerLoopRepeat, bfsInnerLoopStep, bfsOuterLoopCond, bfsOuterLoopRepeat, bfsOuterLoopStep, black, blackTarget, breadthFirstVisit, discoverVertex, edgeIterEnd, edgeIterNext, edgeIterUnpack, emptyECMap, emptyPriorityQueue, emptyVCMap, emptyVPMap, emptyVertexVector, examineEdge, examineVertex, finishVertex, forIterationEnd, front, get, getEdgeCostMap, getVertexCostMap, getVertexPredecessorMap, gray, grayTarget, initMap, isEmptyQueue, iterRangeBegin, iterRangeEnd, less, makeEdge, makeEdgeIteratorRange, makeStateWithMaps, nonTreeEdge, numVertices, outDegree, outEdges, plus, pop, populateVPMapLoopRepeat, populateVPMapLoopStep, primMinimumSpanningTree, push, pushBack, put, putVertexCostMap, putVertexPredecessorMap, relax, second, src, tgt, toEdgeDescriptor, toVertexDescriptor, treeEdge, update, vertexIterEnd, vertexIterNext, vertexIterUnpack, vertices, white)
