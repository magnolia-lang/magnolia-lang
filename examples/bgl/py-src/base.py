from collections import defaultdict, namedtuple
from copy import copy, deepcopy
from enum import Enum
import itertools
import math
from typing import List

import heapq

def base_types():
    class Int: # int
        val: int
        def __init__(self, val):
            self.val = val

        def mutate(self, i2):
            self.val = i2.val

    # TODO: should the hashable requirement be documented anywhere?
    class Vertex:
        val: int
        def __init__(self, val):
            self.val = val

        def mutate(self, i2):
            self.val = i2.val

        def __hash__(self):
            return self.val

        def __eq__(self, other):
            return self.val == other.val

        def __repr__(self):
            return f'{self.__class__.__name__}({self.val})'

        def __copy__(self):
            return Vertex(self.val)

    base_types_tuple = namedtuple('base_types',
                                  ['Int', 'Vertex'])
    return base_types_tuple(Int, Vertex)

def base_float_ops():
    class Float:
        val: float
        def __init__(self, val):
            self.val = val

        def __add__(self, f2):
            return Float(self.val + f2.val)

        def __lt__(self, f2):
            return self.val < f2.val

        def mutate(self, f2):
            self.val = f2.val

        def __copy__(self):
            return Float(self.val)

        def __repr__(self):
            return f'{self.__class__.__name__}({self.val})'

    def plus(f1: Float, f2: Float) -> Float:
        return f1 + f2

    def less(f1: Float, f2: Float) -> Float:
        return f1 < f2

    def negate(f: Float) -> Float:
        return Float(-f.val)

    def zero() -> Float:
        return Float(0.)

    def inf() -> Float:
        return Float(math.inf)

    base_float_ops_tuple = namedtuple('base_float_ops',
                                      ['Float', 'plus', 'less', 'zero', 'inf',
                                       'negate'])
    return base_float_ops_tuple(Float, plus, less, zero, inf, negate)

def color_marker():
    class Color(Enum):
        WHITE = 1
        GRAY = 2
        BLACK = 3

    def white():
        return Color.WHITE

    def gray():
        return Color.GRAY

    def black():
        return Color.BLACK

    color_marker_tuple = namedtuple('color_marker',
                                    ['Color', 'white', 'gray', 'black'])

    return color_marker_tuple(Color, white, gray, black)

def edge(Vertex):
    class Edge:
        def __init__(self, source: Vertex, target: Vertex):
            self.source, self.target = source, target

        def mutate(self, e2):
            self.source, self.target = e2.source, e2.target

        def __hash__(self):
            return hash((self.source, self.target))

        def __eq__(self, other):
            return self.source == other.source and self.target == other.target

    def src(e: Edge) -> Vertex:
        return e.source

    def tgt(e: Edge) -> Vertex:
        return e.target

    def makeEdge(s: Vertex, t: Vertex) -> Edge:
        return Edge(s, t)

    edge_tuple = namedtuple('edge',
                            ['Vertex', 'Edge', 'src', 'tgt', 'makeEdge'])
    return edge_tuple(Vertex, Edge, src, tgt, makeEdge)

def incidence_and_vertex_list_and_edge_list_graph(Vertex):
    class Edge:
        def __init__(self, source, target):
            self.source, self.target = source, target

        def mutate(self, other):
            self.source, self.target = other.source, other.target

    class EdgeDescriptor:
        def __init__(self, source, target):
            self.source, self.target = source, target

        def mutate(self, other):
            self.source, self.target = other.source, other.target

        def __eq__(self, other):
            return self.source == other.source and self.target == other.target

        def __hash__(self):
            return hash((self.source, self.target))

        def __repr__(self):
            return f'EdgeDescriptor({self.source}, {self.target})'

    class EdgeIterator:
        def __init__(self, *args):
            if len(args) == 0:
                pass
            else:
                itr = args[0]
                self.itr = itr
                self.is_end = False
                self.value = None
                self.next()

        def __copy__(self):
            cls = self.__class__
            self.itr, copy_itr = itertools.tee(self.itr)
            result = EdgeIterator()
            result.itr = copy_itr
            result.value = self.value
            result.is_end = self.is_end
            return result

        def mutate(self, other):
            self.itr = copy(other.itr)
            self.is_end = other.is_end
            self.value = other.value

        def unpack(self):
            assert not self.is_end, "can't unpack ended iterator"
            return self.value

        def next(self):
            try:
                self.value = next(self.itr)
            except:
                self.is_end = True
                self.value = None

    class OutEdgeIterator:
        def __init__(self, *args):
            if len(args) == 0:
                pass
            else:
                itr = args[0]
                self.itr = itr
                self.is_end = False
                self.value = None
                self.next()

        def __copy__(self):
            cls = self.__class__
            self.itr, copy_itr = itertools.tee(self.itr)
            result = OutEdgeIterator()
            result.itr = copy_itr
            result.value = self.value
            result.is_end = self.is_end
            return result

        def mutate(self, other):
            self.itr = copy(other.itr)
            self.is_end = other.is_end
            self.value = other.value

        def unpack(self):
            assert not self.is_end, "can't unpack ended iterator"
            return self.value

        def next(self):
            try:
                self.value = next(self.itr)
            except:
                self.is_end = True
                self.value = None

    class VertexDescriptor:
        def __init__(self, vertex):
            self.vertex = vertex

        def mutate(self, other):
            self.vertex = other.vertex

        def __eq__(self, other):
            return self.vertex == other.vertex

        def __hash__(self):
            return self.vertex.__hash__()

        def __repr__(self):
            return f'VertexDescriptor({self.vertex})'

    class VertexIterator:
        def __init__(self, *args): #itr):
            if len(args) == 0:
                pass
            else:
                itr = args[0]
                self.itr = itr
                self.is_end = False
                self.value = None
                self.next()

        def mutate(self, other):
            self.itr = copy(other.itr)
            self.is_end = other.is_end
            self.value = other.value

        def unpack(self):
            assert not self.is_end, "can't unpack ended iterator"
            return self.value

        def next(self):
            try:
                self.value = next(self.itr)
            except:
                self.is_end = True
                self.value = None

        def __copy__(self):
            cls = self.__class__
            self.itr, copy_itr = itertools.tee(self.itr)
            result = VertexIterator()
            result.itr = copy_itr
            result.value = self.value
            result.is_end = self.is_end
            return result

    class VertexCount:
        def __init__(self, val):
            self.val = val

        def mutate(self, other):
            self.val = other.val

    class Graph:
        def __init__(self, edge_list: List[Edge]):
            # TODO: check if vertices are hashable, but for the moment (and the
            #       poc), we don't care.
            self.vertices = set()
            self.out_edge_map = defaultdict(list)
            self.all_edges = []
            for edge in edge_list:
                source, target = (self.toVertexDescriptor(edge.source),
                                  self.toVertexDescriptor(edge.target))
                self.vertices.add(source)
                self.vertices.add(target)
                edge_descriptor = self.toEdgeDescriptor(source, target)
                self.out_edge_map[source].append(edge_descriptor)
                self.all_edges.append(edge_descriptor)

        def toVertexDescriptor(self, v: Vertex):
            return VertexDescriptor(v)

        def toEdgeDescriptor(self, v1: VertexDescriptor, v2: VertexDescriptor):
            return EdgeDescriptor(v1, v2)

        def out_edges(self, v: VertexDescriptor):
            return OutEdgeIterator(iter(self.out_edge_map[v]))

        def num_vertices(self):
            return len(self.vertices)

        def get_edges(self):
            return EdgeIterator(iter(self.all_edges))

        def get_vertices(self):
            return VertexIterator(iter(self.vertices))

    def toEdgeDescriptor(v1: VertexDescriptor, v2: VertexDescriptor,
                         g: Graph) -> EdgeDescriptor:
        return g.toEdgeDescriptor(v1, v2)

    def toVertexDescriptor(v: Vertex, g: Graph) -> VertexDescriptor:
        return g.toVertexDescriptor(v)

    def makeEdge(v1: Vertex, v2: Vertex):
        return Edge(v1, v2)

    def src(ed: EdgeDescriptor, g: Graph) -> VertexDescriptor:
        return ed.source

    def tgt(ed: EdgeDescriptor, g: Graph) -> VertexDescriptor:
        return ed.target

    def edgeIterEnd(itr: EdgeIterator) -> bool:
        return itr.is_end

    def edgeIterNext(itr: EdgeIterator):
        itr.next()

    def edgeIterUnpack(itr: EdgeIterator) -> EdgeDescriptor:
        return itr.unpack()

    def outEdgeIterEnd(itr: OutEdgeIterator) -> bool:
        return itr.is_end

    def outEdgeIterNext(itr: OutEdgeIterator):
        itr.next()

    def outEdgeIterUnpack(itr: OutEdgeIterator) -> EdgeDescriptor:
        return itr.unpack()

    def vertexIterEnd(itr: VertexIterator) -> bool:
        return itr.is_end

    def vertexIterNext(itr: VertexIterator):
        itr.next()

    def vertexIterUnpack(itr: VertexIterator) -> VertexDescriptor:
        return itr.unpack()

    def outEdges(v: VertexDescriptor, g: Graph, ei: OutEdgeIterator):
        ei.mutate(g.out_edges(v))

    def outDegree(v: VertexDescriptor, g: Graph):
        return 0 # TODO

    def edges(g: Graph, itr: EdgeIterator):
        itr.mutate(g.get_edges())

    def vertices(g: Graph, itr: VertexIterator):
        itr.mutate(g.get_vertices())

    def numVertices(g: Graph):
        return g.num_vertices

    incidence_and_vertex_list_and_edge_list_graph_tuple = (
        namedtuple('incidence_and_vertex_list_graph',
                   ['Edge', 'EdgeDescriptor', 'EdgeIterator', 'Graph',
                    'OutEdgeIterator', 'Vertex', 'VertexCount',
                    'VertexDescriptor', 'VertexIterator', 'edgeIterEnd',
                    'edgeIterNext', 'edgeIterUnpack', 'edges',
                    'makeEdge', 'outDegree', 'outEdgeIterEnd',
                    'outEdgeIterNext', 'outEdgeIterUnpack', 'outEdges',
                    'src', 'tgt', 'toEdgeDescriptor', 'toVertexDescriptor',
                    'vertexIterEnd', 'vertexIterNext', 'vertexIterUnpack',
                    'vertices']))

    return incidence_and_vertex_list_and_edge_list_graph_tuple(
        Edge, EdgeDescriptor, EdgeIterator, Graph, OutEdgeIterator, Vertex,
        VertexCount, VertexDescriptor, VertexIterator,
        edgeIterEnd, edgeIterNext, edgeIterUnpack, edges, makeEdge, outDegree,
        outEdgeIterEnd, outEdgeIterNext, outEdgeIterUnpack, outEdges, src, tgt,
        toEdgeDescriptor, toVertexDescriptor, vertexIterEnd, vertexIterNext,
        vertexIterUnpack, vertices)

def custom_incidence_and_vertex_list_graph(Edge, EdgeList, Vertex, VertexList,
        consEdgeList, consVertexList, emptyEdgeList, emptyVertexList,
        headEdgeList, headVertexList, isEmptyEdgeList, isEmptyVertexList,
        makeEdge, src, tailEdgeList, tailVertexList, tgt):

    class VertexCount:
        def __init__(self, count):
            self.count = count

        def __add__(self, vc):
            return VertexCount(self.count + vc2.count)

        def mutate(self, vc):
            self.count = copy(vc2.count)

    class Graph:
        def __init__(self, edges: EdgeList):
            self.edges = []
            self.vertices = set()
            while not isEmptyEdgeList(edges):
                edge = headEdgeList(edges)
                self.vertices.add(src(edge))
                self.vertices.add(tgt(edge))
                self.edges.append(edge)
                edges = tailEdgeList(edges)

        def mutate(self, g):
            self.edges = copy(g2.edges)
            self.vertices = copy(g2.vertices)

    def outEdges(v: Vertex, g: Graph) -> EdgeList:
        result = emptyEdgeList()
        for edge in g.edges:
            if src(edge) == v:
                result = consEdgeList(edge, result)
        return result

    def outDegree(v: Vertex, g: Graph) -> VertexCount:
        outEdgesList = outEdges(v, g)
        result = VertexCount(0)
        while not isEmptyEdgeList(outEdgesList):
            outEdgesList = tailEdgeList(outEdgesList)
            result += VertexCount(1)
        return result

    def vertices(g: Graph) -> VertexList:
        result = emptyVertexList()
        for v in g.vertices:
            result = consVertexList(v, result)
        return result

    def numVertices(g: Graph) -> VertexCount:
        return VertexCount(len(g.vertices))

    custom_incidence_and_vertex_list_graph_tuple = (
        namedtuple('custom_incidence_and_vertex_list_graph',
                   ['Edge', 'EdgeList', 'Graph', 'Vertex', 'VertexCount',
                    'VertexList', 'consEdgeList', 'consVertexList',
                    'emptyEdgeList', 'emptyVertexList', 'headEdgeList',
                    'headVertexList', 'isEmptyEdgeList', 'isEmptyVertexList',
                    'makeEdge', 'src', 'tailEdgeList', 'tailVertexList', 'tgt',
                    'outEdges', 'outDegree', 'vertices', 'numVertices']))

    return custom_incidence_and_vertex_list_graph_tuple(
        Edge, EdgeList, Graph, Vertex, VertexCount, VertexList, consEdgeList,
        consVertexList, emptyEdgeList, emptyVertexList, headEdgeList,
        headVertexList, isEmptyEdgeList, isEmptyVertexList, makeEdge, src,
        tailEdgeList, tailVertexList, tgt, outEdges, outDegree, vertices,
        numVertices)

def list_py(A):
    class List:
        def __init__(self):
            self.list = []

        def cons(self, a: A):
            self.list.insert(0, a)
            return self

        def head(self):
            return self.list[0]

        def tail(self):
            self.list = self.list[1:]
            return self

        def isEmpty(self):
            return not self.list

        def mutate(self, other):
            self.list = other.list[:]

    def cons(a: A, l: List) -> List:
        _l = copy(l)
        return _l.cons(a)

    def empty() -> List:
        return List()

    def head(l: List) -> A:
        return copy(l.head())

    def tail(l: List) -> A:
        return copy(l.tail())

    def isEmpty(l: List) -> bool:
        return l.isEmpty()

    list_py_tuple = namedtuple('list_py',
                               ['A', 'List', 'cons', 'empty', 'head', 'tail',
                                'isEmpty'])
    return list_py_tuple(A, List, cons, empty, head, tail, isEmpty)

def vector(A):
    class Vector:
        def __init__(self):
            self.vector = []

        def push_back(self, a: A):
            self.vector.append(a)

    def empty():
        return Vector()

    def pushBack(a: A, v: Vector) -> Vector:
        v.push_back(a)

    vector_tuple = namedtuple('vector', ['A', 'Vector', 'empty', 'pushBack'])
    return vector_tuple(A, Vector, empty, pushBack)

def read_write_property_map(Key, KeyListIterator, Value,
    iterEnd, iterNext, iterUnpack):

    class PropertyMap:
        def __init__(self, *args):
            if len(args) == 0:
                self.map = dict()
            else:
                self.map = args[0]

        def get(self, k: Key):
            return self.map[k]

        def put(self, k: Key, v: Value):
            self.map[k] = deepcopy(v)

        @classmethod
        def emptyMap(cls):
            return cls(dict())

        @classmethod
        def initMap(cls, itr: KeyListIterator, v: Value):
            dic = dict()
            itr = copy(itr) #itertools.tee(itr)
            while not iterEnd(itr):
                dic[iterUnpack(itr)] = v
                iterNext(itr)
            return cls(dic) #{k: v for k in kl})

        def mutate(self, pm2):
            newMap = dict()
            for k in pm2.map.keys():
                newMap[k] = copy(pm2.map[k])
            self.map = newMap

        def __copy__(self):
            newMap = dict()
            for k in self.map.keys():
                newMap[k] = copy(self.map[k])
            return PropertyMap(newMap)

        def __repr__(self):
            return str(self.map)

    def get(pm: PropertyMap, k: Key):
        return copy(pm.get(k))

    def put(pm: PropertyMap, k: Key, v: Value):
        pm.put(k, v)

    read_write_property_map_tuple = (
        namedtuple('read_write_property_map',
                   ['Key', 'KeyListIterator', 'PropertyMap', 'Value',
                    'emptyMap', 'get', 'put', 'initMap']))
    return read_write_property_map_tuple(Key, KeyListIterator, PropertyMap,
        Value, PropertyMap.emptyMap, get, put, PropertyMap.initMap)

def two_bit_color_map(Key, KeyListIterator, iterEnd, iterNext, iterUnpack):

    class Color(Enum):
        WHITE = 1
        GRAY = 2
        BLACK = 3

    def white():
        return Color.WHITE

    def gray():
        return Color.GRAY

    def black():
        return Color.BLACK

    class ColorPropertyMap:
        def __init__(self):
            self.map = dict()

        def get(self, key):
            return self.map[key]

        def put(self, key, color):
            self.map[key] = color

    def get(cm: ColorPropertyMap, k: Key):
        return cm.get(k)

    def put(cm: ColorPropertyMap, k: Key, c: Color):
        cm.put(k, c)

    def initMap(itr: KeyListIterator, c: Color) -> ColorPropertyMap:
        result = ColorPropertyMap()
        while not iterEnd(itr):
            put(result, iterUnpack(itr), c)
            iterNext(itr)
        return result

    two_bit_color_map_tuple = namedtuple('two_bit_color_map',
        ['Color', 'ColorPropertyMap', 'Key', 'KeyListIterator', 'get',
         'put', 'gray', 'white', 'black', 'initMap'])

    return two_bit_color_map_tuple(Color, ColorPropertyMap, Key,
        KeyListIterator, get, put, gray, white, black, initMap)

def fifo_queue(A):
    class FIFOQueue:
        def __init__(self):
            self.queue = []

        def isEmpty(self):
            return not self.queue

        def push(self, a: A):
            self.queue.append(a)

        def pop(self):
            self.queue = self.queue[1:]

        def front(self):
            return copy(self.queue[0])

        def mutate(self, fq2):
            self.queue = copy(fq2.queue)

    def empty() -> FIFOQueue:
        return FIFOQueue()

    def isEmpty(q: FIFOQueue) -> bool:
        return q.isEmpty()

    def pop(q: FIFOQueue):
        q.pop()

    def push(a: A, q: FIFOQueue) -> FIFOQueue:
        q.push(a)

    def front(q: FIFOQueue) -> A:
        return q.front()

    fifo_queue_tuple = namedtuple('fifo_queue',
                                  ['A', 'FIFOQueue', 'empty', 'isEmpty',
                                   'pop', 'push', 'front'])

    return fifo_queue_tuple(A, FIFOQueue, empty, isEmpty, pop, push, front)


def stack(A):
    class Stack:
        def __init__(self):
            self.stack = []

        def isEmpty(self):
            return not self.stack

        def push(self, a: A):
            self.stack.insert(0, a)

        def pop(self):
            self.stack = self.stack[1:]

        def top(self):
            return copy(self.stack[0])

    def empty():
        return Stack()

    def isEmpty(s: Stack):
        return s.isEmpty()

    def push(a: A, s: Stack):
        s.push(a)

    def pop(s: Stack):
        s.pop()

    def top(s: Stack):
        return s.top()

    stack_tuple = namedtuple('stack',
        ['A', 'Stack', 'empty', 'isEmpty', 'push', 'pop', 'top'])

    return stack_tuple(A, Stack, empty, isEmpty, push, pop, top)


def priority_queue(A, Priority, PriorityMap, get):
    class HeapNode:
        def __init__(self, element, prio):
            self.element, self.prio = element, prio

        def __lt__(self, hn2):
            return self.prio < hn2.prio

        def mutate(self, hn2):
            self.element, self.prio = hn2.element, hn2.prio

    class PriorityQueue:
        def __init__(self, pmap: PriorityMap):
            self.pmap = pmap
            self.heap = []

        def isEmpty(self):
            return not self.heap

        def push(self, a: A):
            heapq.heappush(self.heap, HeapNode(a, get(self.pmap, a)))
            return self

        def pop(self):
            heapq.heappop(self.heap)
            return self

        def front(self):
            return self.heap[0].element

        def update(self, pmap2: PriorityMap, a: A):
            self.pmap = pmap2
            self.heap = list(filter(lambda hn: hn.element != a, self.heap))
            heapq.heapify(self.heap)
            self.push(a)
            return self

        def mutate(self, pq2):
            self.pmap, self.heap = copy(pq2.pmap), copy(pq2.heap)

    def empty(pm: PriorityMap) -> PriorityQueue:
        return PriorityQueue(pm)

    def isEmpty(pq: PriorityQueue) -> bool:
        return pq.isEmpty()

    def update(pm: PriorityMap, a: A, pq: PriorityQueue) -> PriorityQueue:
        _pq = copy(pq)
        _pq.update(pm, a)
        return _pq

    def push(a: A, pq: PriorityQueue) -> PriorityQueue:
        _pq = copy(pq)
        _pq.push(a)
        return _pq

    def pop(pq: PriorityQueue) -> PriorityQueue:
        _pq = copy(pq)
        return _pq.pop()

    def front(pq: PriorityQueue) -> PriorityQueue:
        return copy(pq.front())

    priority_queue_tuple = namedtuple('priority_queue',
                                      ['A', 'Priority', 'PriorityMap',
                                       'PriorityQueue', 'get', 'empty',
                                       'isEmpty', 'update', 'push', 'pop',
                                       'front'])

    return priority_queue_tuple(A, Priority, PriorityMap, PriorityQueue, get,
                                empty, isEmpty, update, push, pop, front)

def pair(A, B):
    class Pair:
        def __init__(self, first: A, second: B):
            self.first, self.second = first, second

        def mutate(self, p2):
            self.first, self.second = p2.first, p2.second

    def first(p: Pair) -> A:
        return p.first

    def second(p: Pair) -> B:
        return p.second

    def makePair(a: A, b: B) -> Pair:
        return Pair(a, b)

    pair_tuple = namedtuple('pair',
                            ['A', 'B', 'Pair', 'first', 'second',
                             'makePair'])
    return pair_tuple(A, B, Pair, first, second, makePair)

def triplet(A, B, C):
    class Triplet:
        def __init__(self, first: A, second: B, third: C):
            self.first, self.second, self.third = first, second, third

        def mutate(self, t2):
            self.first, self.second, self.third = t2.first, t2.second, t2.third

    def first(t: Triplet) -> A:
        return t.first

    def second(t: Triplet) -> B:
        return t.second

    def third(t: Triplet) -> C:
        return t.third

    # TODO: this could just be a constructor. Also for makePair.
    def makeTriplet(a: A, b: B, c: C) -> Triplet:
        return Triplet(a, b, c)

    triplet_tuple = namedtuple('triplet',
                               ['A', 'B', 'C', 'Triplet', 'first', 'second',
                                'third', 'makeTriplet'])
    return triplet_tuple(A, B, C, Triplet, first, second, third, makeTriplet)


def for_iterator_loop(Context, Iterator, State, iterEnd, iterNext, step):

    def forLoopRepeat(in_itr: Iterator, s: State, ctx: Context):
        itr = copy(in_itr)
        while not iterEnd(itr):
            step(itr, s, ctx)
            iterNext(itr)

    for_iterator_loop_tuple = namedtuple('for_iterator_loop',
                            ['Context', 'Iterator', 'State', 'iterEnd',
                             'iterNext', 'step', 'forLoopRepeat'])

    return for_iterator_loop_tuple(Context, Iterator, State, iterEnd, iterNext,
                                   step, forLoopRepeat)

def for_iterator_loop1_2(Context1, Context2, Iterator,
                         State, iterEnd, iterNext, step):

    def forLoopRepeat(in_itr: Iterator, s: State,
                      ctx1: Context1, ctx2: Context2):
        itr = copy(in_itr)
        while not iterEnd(itr):
            step(itr, s, ctx1, ctx2)
            iterNext(itr)

    for_iterator_loop1_2_tuple = namedtuple('for_iterator_loop1_2',
        ['Context1', 'Context2', 'Iterator',
         'State', 'iterEnd', 'iterNext', 'step',
         'forLoopRepeat'])

    return for_iterator_loop1_2_tuple(Context1, Context2, Iterator,
        State, iterEnd, iterNext, step, forLoopRepeat)

def for_iterator_loop1_3(Context1, Context2, Context3, Iterator,
                         State, iterEnd, iterNext, step):

    def forLoopRepeat(in_itr: Iterator, s: State,
                      ctx1: Context1, ctx2: Context2, ctx3: Context3):
        itr = copy(in_itr)
        while not iterEnd(itr):
            step(itr, s, ctx1, ctx2, ctx3)
            iterNext(itr)

    for_iterator_loop1_3_tuple = namedtuple('for_iterator_loop1_3',
        ['Context1', 'Context2', 'Context3', 'Iterator',
         'State', 'iterEnd', 'iterNext', 'step',
         'forLoopRepeat'])

    return for_iterator_loop1_3_tuple(Context1, Context2, Context3, Iterator,
        State, iterEnd, iterNext, step, forLoopRepeat)

def for_iterator_loop2_3(Context1, Context2, Context3, Iterator,
                         State1, State2, iterEnd, iterNext, step):

    def forLoopRepeat(in_itr: Iterator, s1: State1, s2: State2,
                      ctx1: Context1, ctx2: Context2, ctx3: Context3):
        itr = copy(in_itr)
        while not iterEnd(itr):
            step(itr, s1, s2, ctx1, ctx2, ctx3)
            iterNext(itr)

    for_iterator_loop2_3_tuple = namedtuple('for_iterator_loop2_3',
        ['Context1', 'Context2', 'Context3', 'Iterator',
         'State1', 'State2', 'iterEnd', 'iterNext', 'step',
         'forLoopRepeat'])

    return for_iterator_loop2_3_tuple(Context1, Context2, Context3, Iterator,
        State1, State2, iterEnd, iterNext, step, forLoopRepeat)


def for_iterator_loop3_2(Context1, Context2, Iterator,
                         State1, State2, State3, iterEnd, iterNext, step):

    def forLoopRepeat(in_itr: Iterator, s1: State1, s2: State2, s3: State3,
                      ctx1: Context1, ctx2: Context2):
        itr = copy(in_itr)
        while not iterEnd(itr):
            step(itr, s1, s2, s3, ctx1, ctx2)
            iterNext(itr)

    for_iterator_loop3_2_tuple = namedtuple('for_iterator_loop3_2',
        ['Context1', 'Context2', 'Iterator',
         'State1', 'State2', 'State3', 'iterEnd', 'iterNext', 'step',
         'forLoopRepeat'])

    return for_iterator_loop3_2_tuple(Context1, Context2, Iterator,
        State1, State2, State3, iterEnd, iterNext, step, forLoopRepeat)

# TODO: not really parallel, but here for sake of completion
def for_parallel_iterator_loop3_2(Context1, Context2, Iterator,
                         State1, State2, State3, iterEnd, iterNext, step):

    def forLoopRepeat(in_itr: Iterator, s1: State1, s2: State2, s3: State3,
                      ctx1: Context1, ctx2: Context2):
        itr = copy(in_itr)
        while not iterEnd(itr):
            step(itr, s1, s2, s3, ctx1, ctx2)
            iterNext(itr)

    for_iterator_loop3_2_tuple('for_iterator_loop3_2',
        ['Context1', 'Context2', 'Iterator',
         'State1', 'State2', 'State3', 'iterEnd', 'iterNext', 'step'])

    return for_iterator_loop_tuple(Context1, Context2, Iterator,
        State1, State2, State3, iterEnd, iterNext, step)

def while_loop(Context, State, cond, step):
    def repeat(state: State, context: Context):
        while cond(state, context):
            step(state, context)

    while_loop_tuple = namedtuple('while_loop',
                                  ['Context', 'State', 'cond', 'step',
                                   'repeat'])
    return while_loop_tuple(Context, State, cond, step, repeat)

def while_loop3(Context, State1, State2, State3, cond, step):
    def repeat(s1: State1, s2: State2, s3: State3, context: Context):
        while cond(s1, s2, s3, context):
            step(s1, s2, s3, context)

    while_loop_tuple = namedtuple('while_loop',
        ['Context', 'State1', 'State2', 'State3', 'cond', 'step', 'repeat'])
    return while_loop_tuple(Context, State1, State2, State3, cond, step, repeat)

def base_bool():
    class Bool:
        val: bool

        def __init__(self, *args):
            if len(args) == 0:
                self.val = False
            else:
                self.val = args[0]

        def mutate(self, other):
            self.val = other.val

        def __eq__(self, other):
            return self.val == other.val

    def bfalse():
        return Bool(False)

    def btrue():
        return Bool(True)

    base_bool_tuple = namedtuple('base_bool', ['Bool', 'bfalse', 'btrue'])
    return base_bool_tuple(Bool, bfalse, btrue)

def base_unit():
    class Unit:
        def __init__(self):
            pass

        def __eq__(self, other):
            return True

    def unit():
        return Unit()

    base_unit_tuple = namedtuple('base_tuple', ['Unit', 'unit'])
    return base_unit_tuple(Unit, unit)

