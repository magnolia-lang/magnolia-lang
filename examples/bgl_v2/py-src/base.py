from collections import namedtuple
from copy import copy
from enum import Enum
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

        def __repr__(self):
            return f'{self.__class__.__name__}({self.val})'
    
    def plus(f1: Float, f2: Float) -> Float:
        return f1 + f2

    def less(f1: Float, f2: Float) -> Float:
        return f1 < f2

    base_float_ops_tuple = namedtuple('base_float_ops',
                                      ['Float', 'plus', 'less'])
    return base_float_ops_tuple(Float, plus, less)

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

def incidence_and_vertex_list_graph(Edge, EdgeList, Vertex, VertexList,
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
    
    incidence_and_vertex_list_graph_tuple = (
        namedtuple('incidence_and_vertex_list_graph',
                   ['Edge', 'EdgeList', 'Graph', 'Vertex', 'VertexCount',
                    'VertexList', 'consEdgeList', 'consVertexList',
                    'emptyEdgeList', 'emptyVertexList', 'headEdgeList',
                    'headVertexList', 'isEmptyEdgeList', 'isEmptyVertexList',
                    'makeEdge', 'src', 'tailEdgeList', 'tailVertexList', 'tgt',
                    'outEdges', 'outDegree', 'vertices', 'numVertices']))
    
    return incidence_and_vertex_list_graph_tuple(
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

def read_write_property_map(Key, KeyList, Value, cons, emptyKeyList,
                            head, isEmpty, tail):
    class PropertyMap:
        def __init__(self, _map):
            self.map = _map

        def get(self, k: Key):
            return self.map[k]

        def put(self, k: Key, v: Value):
            self.map[k] = v
            return self

        @classmethod
        def emptyMap(cls):
            return cls(dict())

        @classmethod
        def initMap(cls, kl: KeyList, v: Value):
            dic = dict()
            while not isEmpty(kl):
                dic[head(kl)] = v
                kl = tail(kl)
            return cls(dic) #{k: v for k in kl})

        def mutate(self, pm2):
            self.map = copy(pm2.map)

        def __repr__(self):
            return str(self.map)
    def get(pm: PropertyMap, k: Key):
        return copy(pm.get(k))

    def put(pm: PropertyMap, k: Key, v: Value):
        _pm = copy(pm)
        _pm.put(k, v)
        return _pm

    read_write_property_map_tuple = (
        namedtuple('read_write_property_map',
                   ['Key', 'KeyList', 'PropertyMap', 'Value', 'cons',
                    'emptyKeyList', 'head', 'isEmpty', 'tail', 'emptyMap',
                    'get', 'put', 'initMap']))
    return read_write_property_map_tuple(Key, KeyList, PropertyMap, Value, cons,
        emptyKeyList, head, isEmpty, tail, PropertyMap.emptyMap, get, put,
        PropertyMap.initMap)

def fifo_queue(A):
    class FIFOQueue:
        def __init__(self):
            self.queue = []

        def isEmpty(self):
            return not self.queue

        def push(self, a: A):
            self.queue.append(a)
            return self

        def pop(self):
            self.queue = self.queue[1:]
            return self

        def front(self):
            return self.queue[0]

        def mutate(self, fq2):
            self.queue = copy(fq2.queue)

    def empty() -> FIFOQueue:
        return FIFOQueue()

    def isEmpty(q: FIFOQueue) -> bool:
        return q.isEmpty()

    def pop(q: FIFOQueue) -> FIFOQueue:
        _q = copy(q)
        return _q.pop()

    def push(a: A, q: FIFOQueue) -> FIFOQueue:
        _q = copy(q)
        return _q.push(a)

    def front(q: FIFOQueue) -> A:
        return copy(q.front())

    fifo_queue_tuple = namedtuple('fifo_queue',
                                  ['A', 'FIFOQueue', 'empty', 'isEmpty',
                                   'pop', 'push', 'front'])

    return fifo_queue_tuple(A, FIFOQueue, empty, isEmpty, pop, push, front)

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


def while_loop(Context, State, cond, step):
    def repeat(state: State, context: Context):
        while cond(state, context):
            step(state, context)

    while_loop_tuple = namedtuple('while_loop',
                                  ['Context', 'State', 'cond', 'step',
                                   'repeat'])
    return while_loop_tuple(Context, State, cond, step, repeat)
