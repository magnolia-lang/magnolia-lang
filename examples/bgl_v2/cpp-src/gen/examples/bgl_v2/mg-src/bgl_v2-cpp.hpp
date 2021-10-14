#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
struct CppBFSTestVisitor {
private:
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::VertexDescriptor VertexDescriptor;
    typedef vector<CppBFSTestVisitor::VertexDescriptor>::Vector VertexVector;
    struct _emptyVertexVector {
        inline CppBFSTestVisitor::VertexVector operator()() {
            return __vector.empty();
        };
    };

    static CppBFSTestVisitor::_emptyVertexVector emptyVertexVector;
private:
    static fifo_queue<CppBFSTestVisitor::VertexDescriptor> __fifo_queue;
    static vector<CppBFSTestVisitor::VertexDescriptor> __vector;
public:
    struct _pushBack {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& a, CppBFSTestVisitor::VertexVector& v) {
            return __vector.pushBack(a, v);
        };
    };

    static CppBFSTestVisitor::_pushBack pushBack;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterEnd(ei);
        };
    };

    static CppBFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterNext(ei);
        };
    };

    static CppBFSTestVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterUnpack(ei);
        };
    };

private:
    static two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterEnd, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppBFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex> __incidence_and_vertex_list_graph;
public:
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::Graph Graph;
    struct _breadthFirstSearch {
        inline CppBFSTestVisitor::VertexVector operator()(const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::VertexDescriptor& start, const CppBFSTestVisitor::VertexVector& init) {
            CppBFSTestVisitor::FIFOQueue q = CppBFSTestVisitor::empty();
            CppBFSTestVisitor::VertexIterator vertexItr;
            CppBFSTestVisitor::vertices(g, vertexItr);
            CppBFSTestVisitor::ColorPropertyMap c = CppBFSTestVisitor::initMap(vertexItr, CppBFSTestVisitor::white());
            CppBFSTestVisitor::VertexVector a = init;
            CppBFSTestVisitor::breadthFirstVisit(g, start, a, q, c);
            return a;
        };
    };

    static CppBFSTestVisitor::_breadthFirstSearch breadthFirstSearch;
    struct _numVertices {
        inline CppBFSTestVisitor::VertexCount operator()(const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.numVertices(g);
        };
    };

    static CppBFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppBFSTestVisitor::VertexCount operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.outDegree(v, g);
        };
    };

    static CppBFSTestVisitor::_outDegree outDegree;
    struct _toVertexDescriptor {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppBFSTestVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_graph.vertices(g, itr);
        };
    };

    static CppBFSTestVisitor::_vertices vertices;
    typedef fifo_queue<CppBFSTestVisitor::VertexDescriptor>::FIFOQueue FIFOQueue;
    struct _empty {
        inline CppBFSTestVisitor::FIFOQueue operator()() {
            return __fifo_queue.empty();
        };
    };

    static CppBFSTestVisitor::_empty empty;
    struct _examineVertex {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexVector& a) {
            CppBFSTestVisitor::pushBack(v, a);
        };
    };

    static CppBFSTestVisitor::_examineVertex examineVertex;
    struct _front {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.front(q);
        };
    };

    static CppBFSTestVisitor::_front front;
    struct _isEmptyQueue {
        inline bool operator()(const CppBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.isEmpty(q);
        };
    };

    static CppBFSTestVisitor::_isEmptyQueue isEmptyQueue;
    struct _pop {
        inline void operator()(CppBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.pop(q);
        };
    };

    static CppBFSTestVisitor::_pop pop;
    struct _push {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& a, CppBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.push(a, q);
        };
    };

    static CppBFSTestVisitor::_push push;
    struct _pushPopBehavior {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& a, const CppBFSTestVisitor::FIFOQueue& inq) {
            CppBFSTestVisitor::FIFOQueue mut_inq = inq;
            CppBFSTestVisitor::push(a, mut_inq);
            assert((CppBFSTestVisitor::front(mut_inq)) == (a));
            CppBFSTestVisitor::pop(mut_inq);
            assert((inq) == (mut_inq));
        };
    };

    static CppBFSTestVisitor::_pushPopBehavior pushPopBehavior;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::EdgeIterator EdgeIterator;
    typedef pair<CppBFSTestVisitor::EdgeIterator, CppBFSTestVisitor::EdgeIterator>::Pair EdgeIteratorRange;
private:
    static pair<CppBFSTestVisitor::EdgeIterator, CppBFSTestVisitor::EdgeIterator> __pair;
public:
    struct _edgeIterEnd {
        inline bool operator()(const CppBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterEnd(ei);
        };
    };

    static CppBFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterNext(ei);
        };
    };

    static CppBFSTestVisitor::_edgeIterNext edgeIterNext;
    struct _iterRangeBegin {
        inline CppBFSTestVisitor::EdgeIterator operator()(const CppBFSTestVisitor::EdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppBFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppBFSTestVisitor::EdgeIterator operator()(const CppBFSTestVisitor::EdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppBFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeEdgeIteratorRange {
        inline CppBFSTestVisitor::EdgeIteratorRange operator()(const CppBFSTestVisitor::EdgeIterator& a, const CppBFSTestVisitor::EdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppBFSTestVisitor::_makeEdgeIteratorRange makeEdgeIteratorRange;
    struct _outEdges {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_graph.outEdges(v, g, itr);
        };
    };

    static CppBFSTestVisitor::_outEdges outEdges;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _defaultAction {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& edgeOrVertex, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexVector& a) {
            ;
        };
        inline void operator()(const CppBFSTestVisitor::EdgeDescriptor& edgeOrVertex, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexVector& a) {
            ;
        };
    };

    static CppBFSTestVisitor::_defaultAction defaultAction;
    struct _edgeIterUnpack {
        inline CppBFSTestVisitor::EdgeDescriptor operator()(const CppBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppBFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _src {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::EdgeDescriptor& e, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.src(e, g);
        };
    };

    static CppBFSTestVisitor::_src src;
    struct _tgt {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::EdgeDescriptor& e, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.tgt(e, g);
        };
    };

    static CppBFSTestVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppBFSTestVisitor::EdgeDescriptor operator()(const CppBFSTestVisitor::VertexDescriptor& v1, const CppBFSTestVisitor::VertexDescriptor& v2, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppBFSTestVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_graph<CppBFSTestVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppBFSTestVisitor::Edge operator()(const CppBFSTestVisitor::Vertex& s, const CppBFSTestVisitor::Vertex& t) {
            return __incidence_and_vertex_list_graph.makeEdge(s, t);
        };
    };

    static CppBFSTestVisitor::_makeEdge makeEdge;
    typedef two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterEnd, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppBFSTestVisitor::EdgeIterator& itr, CppBFSTestVisitor::VertexVector& s1, CppBFSTestVisitor::FIFOQueue& s2, CppBFSTestVisitor::ColorPropertyMap& s3, const CppBFSTestVisitor::Graph& ctx1, const CppBFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppBFSTestVisitor::EdgeIterator& edgeItr, CppBFSTestVisitor::VertexVector& x, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c, const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::VertexDescriptor& u) {
            CppBFSTestVisitor::EdgeDescriptor e = CppBFSTestVisitor::edgeIterUnpack(edgeItr);
            CppBFSTestVisitor::VertexDescriptor v = CppBFSTestVisitor::tgt(e, g);
            CppBFSTestVisitor::defaultAction(e, g, q, x);
            CppBFSTestVisitor::Color vc = CppBFSTestVisitor::get(c, v);
            if ((vc) == (CppBFSTestVisitor::white()))
            {
                CppBFSTestVisitor::defaultAction(e, g, q, x);
                CppBFSTestVisitor::put(c, v, CppBFSTestVisitor::gray());
                CppBFSTestVisitor::defaultAction(v, g, q, x);
                CppBFSTestVisitor::push(v, q);
            }
            else
                if ((vc) == (CppBFSTestVisitor::gray()))
                {
                    CppBFSTestVisitor::defaultAction(e, g, q, x);
                }
                else
                {
                    CppBFSTestVisitor::defaultAction(e, g, q, x);
                }
        };
    };

private:
    static for_iterator_loop3_2<CppBFSTestVisitor::Graph, CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::EdgeIterator, CppBFSTestVisitor::VertexVector, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap, CppBFSTestVisitor::_edgeIterEnd, CppBFSTestVisitor::_edgeIterNext, CppBFSTestVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
public:
    static CppBFSTestVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
    struct _bfsOuterLoopCond {
        inline bool operator()(const CppBFSTestVisitor::VertexVector& a, const CppBFSTestVisitor::FIFOQueue& q, const CppBFSTestVisitor::ColorPropertyMap& c, const CppBFSTestVisitor::Graph& g) {
            return !CppBFSTestVisitor::isEmptyQueue(q);
        };
    };

    static CppBFSTestVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        inline void operator()(CppBFSTestVisitor::VertexVector& s1, CppBFSTestVisitor::FIFOQueue& s2, CppBFSTestVisitor::ColorPropertyMap& s3, const CppBFSTestVisitor::Graph& ctx) {
            return __while_loop3.repeat(s1, s2, s3, ctx);
        };
    };

    static CppBFSTestVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        inline void operator()(CppBFSTestVisitor::VertexVector& x, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c, const CppBFSTestVisitor::Graph& g) {
            CppBFSTestVisitor::VertexDescriptor u = CppBFSTestVisitor::front(q);
            CppBFSTestVisitor::pop(q);
            CppBFSTestVisitor::examineVertex(u, g, q, x);
            CppBFSTestVisitor::EdgeIterator edgeItr;
            CppBFSTestVisitor::outEdges(u, g, edgeItr);
            CppBFSTestVisitor::bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);
            CppBFSTestVisitor::put(c, u, CppBFSTestVisitor::black());
            CppBFSTestVisitor::defaultAction(u, g, q, x);
        };
    };

private:
    static while_loop3<CppBFSTestVisitor::Graph, CppBFSTestVisitor::VertexVector, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap, CppBFSTestVisitor::_bfsOuterLoopCond, CppBFSTestVisitor::_bfsOuterLoopStep> __while_loop3;
public:
    static CppBFSTestVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _breadthFirstVisit {
        inline void operator()(const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::VertexDescriptor& s, CppBFSTestVisitor::VertexVector& a, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c) {
            CppBFSTestVisitor::defaultAction(s, g, q, a);
            CppBFSTestVisitor::push(s, q);
            CppBFSTestVisitor::put(c, s, CppBFSTestVisitor::gray());
            CppBFSTestVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppBFSTestVisitor::_breadthFirstVisit breadthFirstVisit;
    typedef two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterEnd, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack>::Color Color;
    struct _black {
        inline CppBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.black();
        };
    };

    static CppBFSTestVisitor::_black black;
    struct _get {
        inline CppBFSTestVisitor::Color operator()(const CppBFSTestVisitor::ColorPropertyMap& pm, const CppBFSTestVisitor::VertexDescriptor& k) {
            return __two_bit_color_map.get(pm, k);
        };
    };

    static CppBFSTestVisitor::_get get;
    struct _gray {
        inline CppBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.gray();
        };
    };

    static CppBFSTestVisitor::_gray gray;
    struct _initMap {
        inline CppBFSTestVisitor::ColorPropertyMap operator()(const CppBFSTestVisitor::VertexIterator& kli, const CppBFSTestVisitor::Color& v) {
            return __two_bit_color_map.initMap(kli, v);
        };
    };

    static CppBFSTestVisitor::_initMap initMap;
    struct _put {
        inline void operator()(CppBFSTestVisitor::ColorPropertyMap& pm, const CppBFSTestVisitor::VertexDescriptor& k, const CppBFSTestVisitor::Color& v) {
            return __two_bit_color_map.put(pm, k, v);
        };
    };

    static CppBFSTestVisitor::_put put;
    struct _white {
        inline CppBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.white();
        };
    };

    static CppBFSTestVisitor::_white white;
};
} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp

namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
struct CppDFSTestVisitor {
    struct _emptyStackIsEmpty {
        inline void operator()() {
            assert(CppDFSTestVisitor::isEmptyStack(CppDFSTestVisitor::empty()));
        };
    };

    static CppDFSTestVisitor::_emptyStackIsEmpty emptyStackIsEmpty;
private:
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::VertexDescriptor VertexDescriptor;
    typedef vector<CppDFSTestVisitor::VertexDescriptor>::Vector VertexVector;
    struct _emptyVertexVector {
        inline CppDFSTestVisitor::VertexVector operator()() {
            return __vector.empty();
        };
    };

    static CppDFSTestVisitor::_emptyVertexVector emptyVertexVector;
private:
    static stack<CppDFSTestVisitor::VertexDescriptor> __stack;
    static vector<CppDFSTestVisitor::VertexDescriptor> __vector;
public:
    struct _pushBack {
        inline void operator()(const CppDFSTestVisitor::VertexDescriptor& a, CppDFSTestVisitor::VertexVector& v) {
            return __vector.pushBack(a, v);
        };
    };

    static CppDFSTestVisitor::_pushBack pushBack;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppDFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterEnd(ei);
        };
    };

    static CppDFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppDFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterNext(ei);
        };
    };

    static CppDFSTestVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterUnpack(ei);
        };
    };

private:
    static two_bit_color_map<CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::VertexIterator, CppDFSTestVisitor::_vertexIterEnd, CppDFSTestVisitor::_vertexIterNext, CppDFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppDFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex> __incidence_and_vertex_list_graph;
public:
    typedef stack<CppDFSTestVisitor::VertexDescriptor>::Stack Stack;
    struct _empty {
        inline CppDFSTestVisitor::Stack operator()() {
            return __stack.empty();
        };
    };

    static CppDFSTestVisitor::_empty empty;
    struct _isEmptyStack {
        inline bool operator()(const CppDFSTestVisitor::Stack& s) {
            return __stack.isEmpty(s);
        };
    };

    static CppDFSTestVisitor::_isEmptyStack isEmptyStack;
    struct _pop {
        inline void operator()(CppDFSTestVisitor::Stack& s) {
            return __stack.pop(s);
        };
    };

    static CppDFSTestVisitor::_pop pop;
    struct _push {
        inline void operator()(const CppDFSTestVisitor::VertexDescriptor& a, CppDFSTestVisitor::Stack& s) {
            return __stack.push(a, s);
        };
    };

    static CppDFSTestVisitor::_push push;
    struct _pushPopTopBehavior {
        inline void operator()(const CppDFSTestVisitor::Stack& s, const CppDFSTestVisitor::VertexDescriptor& a) {
            CppDFSTestVisitor::Stack mut_s = s;
            CppDFSTestVisitor::push(a, mut_s);
            assert((CppDFSTestVisitor::top(mut_s)) == (a));
            CppDFSTestVisitor::pop(mut_s);
            assert((mut_s) == (s));
        };
    };

    static CppDFSTestVisitor::_pushPopTopBehavior pushPopTopBehavior;
    struct _top {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::Stack& s) {
            return __stack.top(s);
        };
    };

    static CppDFSTestVisitor::_top top;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::Graph Graph;
    struct _depthFirstSearch {
        inline CppDFSTestVisitor::VertexVector operator()(const CppDFSTestVisitor::Graph& g, const CppDFSTestVisitor::VertexDescriptor& start, const CppDFSTestVisitor::VertexVector& init) {
            CppDFSTestVisitor::Stack q = CppDFSTestVisitor::empty();
            CppDFSTestVisitor::VertexIterator vertexItr;
            CppDFSTestVisitor::vertices(g, vertexItr);
            CppDFSTestVisitor::ColorPropertyMap c = CppDFSTestVisitor::initMap(vertexItr, CppDFSTestVisitor::white());
            CppDFSTestVisitor::VertexVector a = init;
            CppDFSTestVisitor::breadthFirstVisit(g, start, a, q, c);
            return a;
        };
    };

    static CppDFSTestVisitor::_depthFirstSearch depthFirstSearch;
    struct _examineVertex {
        inline void operator()(const CppDFSTestVisitor::VertexDescriptor& v, const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::VertexVector& a) {
            CppDFSTestVisitor::pushBack(v, a);
        };
    };

    static CppDFSTestVisitor::_examineVertex examineVertex;
    struct _numVertices {
        inline CppDFSTestVisitor::VertexCount operator()(const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.numVertices(g);
        };
    };

    static CppDFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppDFSTestVisitor::VertexCount operator()(const CppDFSTestVisitor::VertexDescriptor& v, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.outDegree(v, g);
        };
    };

    static CppDFSTestVisitor::_outDegree outDegree;
    struct _toVertexDescriptor {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::Vertex& v, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppDFSTestVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_graph.vertices(g, itr);
        };
    };

    static CppDFSTestVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::EdgeIterator EdgeIterator;
    typedef pair<CppDFSTestVisitor::EdgeIterator, CppDFSTestVisitor::EdgeIterator>::Pair EdgeIteratorRange;
private:
    static pair<CppDFSTestVisitor::EdgeIterator, CppDFSTestVisitor::EdgeIterator> __pair;
public:
    struct _edgeIterEnd {
        inline bool operator()(const CppDFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterEnd(ei);
        };
    };

    static CppDFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppDFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterNext(ei);
        };
    };

    static CppDFSTestVisitor::_edgeIterNext edgeIterNext;
    struct _iterRangeBegin {
        inline CppDFSTestVisitor::EdgeIterator operator()(const CppDFSTestVisitor::EdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppDFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppDFSTestVisitor::EdgeIterator operator()(const CppDFSTestVisitor::EdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppDFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeEdgeIteratorRange {
        inline CppDFSTestVisitor::EdgeIteratorRange operator()(const CppDFSTestVisitor::EdgeIterator& a, const CppDFSTestVisitor::EdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppDFSTestVisitor::_makeEdgeIteratorRange makeEdgeIteratorRange;
    struct _outEdges {
        inline void operator()(const CppDFSTestVisitor::VertexDescriptor& v, const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_graph.outEdges(v, g, itr);
        };
    };

    static CppDFSTestVisitor::_outEdges outEdges;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _defaultAction {
        inline void operator()(const CppDFSTestVisitor::VertexDescriptor& edgeOrVertex, const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::VertexVector& a) {
            ;
        };
        inline void operator()(const CppDFSTestVisitor::EdgeDescriptor& edgeOrVertex, const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::VertexVector& a) {
            ;
        };
    };

    static CppDFSTestVisitor::_defaultAction defaultAction;
    struct _edgeIterUnpack {
        inline CppDFSTestVisitor::EdgeDescriptor operator()(const CppDFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppDFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _src {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::EdgeDescriptor& e, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.src(e, g);
        };
    };

    static CppDFSTestVisitor::_src src;
    struct _tgt {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::EdgeDescriptor& e, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.tgt(e, g);
        };
    };

    static CppDFSTestVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppDFSTestVisitor::EdgeDescriptor operator()(const CppDFSTestVisitor::VertexDescriptor& v1, const CppDFSTestVisitor::VertexDescriptor& v2, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppDFSTestVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_graph<CppDFSTestVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppDFSTestVisitor::Edge operator()(const CppDFSTestVisitor::Vertex& s, const CppDFSTestVisitor::Vertex& t) {
            return __incidence_and_vertex_list_graph.makeEdge(s, t);
        };
    };

    static CppDFSTestVisitor::_makeEdge makeEdge;
    typedef two_bit_color_map<CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::VertexIterator, CppDFSTestVisitor::_vertexIterEnd, CppDFSTestVisitor::_vertexIterNext, CppDFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppDFSTestVisitor::EdgeIterator& itr, CppDFSTestVisitor::VertexVector& s1, CppDFSTestVisitor::Stack& s2, CppDFSTestVisitor::ColorPropertyMap& s3, const CppDFSTestVisitor::Graph& ctx1, const CppDFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppDFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppDFSTestVisitor::EdgeIterator& edgeItr, CppDFSTestVisitor::VertexVector& x, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::ColorPropertyMap& c, const CppDFSTestVisitor::Graph& g, const CppDFSTestVisitor::VertexDescriptor& u) {
            CppDFSTestVisitor::EdgeDescriptor e = CppDFSTestVisitor::edgeIterUnpack(edgeItr);
            CppDFSTestVisitor::VertexDescriptor v = CppDFSTestVisitor::tgt(e, g);
            CppDFSTestVisitor::defaultAction(e, g, q, x);
            CppDFSTestVisitor::Color vc = CppDFSTestVisitor::get(c, v);
            if ((vc) == (CppDFSTestVisitor::white()))
            {
                CppDFSTestVisitor::defaultAction(e, g, q, x);
                CppDFSTestVisitor::put(c, v, CppDFSTestVisitor::gray());
                CppDFSTestVisitor::defaultAction(v, g, q, x);
                CppDFSTestVisitor::push(v, q);
            }
            else
                if ((vc) == (CppDFSTestVisitor::gray()))
                {
                    CppDFSTestVisitor::defaultAction(e, g, q, x);
                }
                else
                {
                    CppDFSTestVisitor::defaultAction(e, g, q, x);
                }
        };
    };

private:
    static for_iterator_loop3_2<CppDFSTestVisitor::Graph, CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::EdgeIterator, CppDFSTestVisitor::VertexVector, CppDFSTestVisitor::Stack, CppDFSTestVisitor::ColorPropertyMap, CppDFSTestVisitor::_edgeIterEnd, CppDFSTestVisitor::_edgeIterNext, CppDFSTestVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
public:
    static CppDFSTestVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
    struct _bfsOuterLoopCond {
        inline bool operator()(const CppDFSTestVisitor::VertexVector& a, const CppDFSTestVisitor::Stack& q, const CppDFSTestVisitor::ColorPropertyMap& c, const CppDFSTestVisitor::Graph& g) {
            return !CppDFSTestVisitor::isEmptyStack(q);
        };
    };

    static CppDFSTestVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        inline void operator()(CppDFSTestVisitor::VertexVector& s1, CppDFSTestVisitor::Stack& s2, CppDFSTestVisitor::ColorPropertyMap& s3, const CppDFSTestVisitor::Graph& ctx) {
            return __while_loop3.repeat(s1, s2, s3, ctx);
        };
    };

    static CppDFSTestVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        inline void operator()(CppDFSTestVisitor::VertexVector& x, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::ColorPropertyMap& c, const CppDFSTestVisitor::Graph& g) {
            CppDFSTestVisitor::VertexDescriptor u = CppDFSTestVisitor::top(q);
            CppDFSTestVisitor::pop(q);
            CppDFSTestVisitor::examineVertex(u, g, q, x);
            CppDFSTestVisitor::EdgeIterator edgeItr;
            CppDFSTestVisitor::outEdges(u, g, edgeItr);
            CppDFSTestVisitor::bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);
            CppDFSTestVisitor::put(c, u, CppDFSTestVisitor::black());
            CppDFSTestVisitor::defaultAction(u, g, q, x);
        };
    };

private:
    static while_loop3<CppDFSTestVisitor::Graph, CppDFSTestVisitor::VertexVector, CppDFSTestVisitor::Stack, CppDFSTestVisitor::ColorPropertyMap, CppDFSTestVisitor::_bfsOuterLoopCond, CppDFSTestVisitor::_bfsOuterLoopStep> __while_loop3;
public:
    static CppDFSTestVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _breadthFirstVisit {
        inline void operator()(const CppDFSTestVisitor::Graph& g, const CppDFSTestVisitor::VertexDescriptor& s, CppDFSTestVisitor::VertexVector& a, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::ColorPropertyMap& c) {
            CppDFSTestVisitor::defaultAction(s, g, q, a);
            CppDFSTestVisitor::push(s, q);
            CppDFSTestVisitor::put(c, s, CppDFSTestVisitor::gray());
            CppDFSTestVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppDFSTestVisitor::_breadthFirstVisit breadthFirstVisit;
    typedef two_bit_color_map<CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::VertexIterator, CppDFSTestVisitor::_vertexIterEnd, CppDFSTestVisitor::_vertexIterNext, CppDFSTestVisitor::_vertexIterUnpack>::Color Color;
    struct _black {
        inline CppDFSTestVisitor::Color operator()() {
            return __two_bit_color_map.black();
        };
    };

    static CppDFSTestVisitor::_black black;
    struct _get {
        inline CppDFSTestVisitor::Color operator()(const CppDFSTestVisitor::ColorPropertyMap& pm, const CppDFSTestVisitor::VertexDescriptor& k) {
            return __two_bit_color_map.get(pm, k);
        };
    };

    static CppDFSTestVisitor::_get get;
    struct _gray {
        inline CppDFSTestVisitor::Color operator()() {
            return __two_bit_color_map.gray();
        };
    };

    static CppDFSTestVisitor::_gray gray;
    struct _initMap {
        inline CppDFSTestVisitor::ColorPropertyMap operator()(const CppDFSTestVisitor::VertexIterator& kli, const CppDFSTestVisitor::Color& v) {
            return __two_bit_color_map.initMap(kli, v);
        };
    };

    static CppDFSTestVisitor::_initMap initMap;
    struct _put {
        inline void operator()(CppDFSTestVisitor::ColorPropertyMap& pm, const CppDFSTestVisitor::VertexDescriptor& k, const CppDFSTestVisitor::Color& v) {
            return __two_bit_color_map.put(pm, k, v);
        };
    };

    static CppDFSTestVisitor::_put put;
    struct _white {
        inline CppDFSTestVisitor::Color operator()() {
            return __two_bit_color_map.white();
        };
    };

    static CppDFSTestVisitor::_white white;
};
} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp

namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
struct CppDijkstraVisitor {
private:
    static base_types __base_types;
    static base_float_ops __base_float_ops;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::VertexDescriptor VertexDescriptor;
    typedef vector<CppDijkstraVisitor::VertexDescriptor>::Vector VertexVector;
    struct _emptyVertexVector {
        inline CppDijkstraVisitor::VertexVector operator()() {
            return __vector.empty();
        };
    };

    static CppDijkstraVisitor::_emptyVertexVector emptyVertexVector;
private:
    static vector<CppDijkstraVisitor::VertexDescriptor> __vector;
public:
    struct _pushBack {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& a, CppDijkstraVisitor::VertexVector& v) {
            return __vector.pushBack(a, v);
        };
    };

    static CppDijkstraVisitor::_pushBack pushBack;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppDijkstraVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterEnd(ei);
        };
    };

    static CppDijkstraVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppDijkstraVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterNext(ei);
        };
    };

    static CppDijkstraVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterUnpack(ei);
        };
    };

    typedef read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::PropertyMap VertexPredecessorMap;
    struct _emptyVPMap {
        inline CppDijkstraVisitor::VertexPredecessorMap operator()() {
            return __read_write_property_map1.emptyMap();
        };
    };

    static CppDijkstraVisitor::_emptyVPMap emptyVPMap;
    struct _forIterationEnd {
        inline void operator()(const CppDijkstraVisitor::VertexIterator& itr, const CppDijkstraVisitor::VertexPredecessorMap& state, const CppDijkstraVisitor::VertexDescriptor& ctx) {
            CppDijkstraVisitor::VertexPredecessorMap mut_state = state;
            if (CppDijkstraVisitor::vertexIterEnd(itr))
            {
                CppDijkstraVisitor::populateVPMapLoopRepeat(itr, mut_state, ctx);
                assert((mut_state) == (state));
            }
            else
                ;
        };
    };

    static CppDijkstraVisitor::_forIterationEnd forIterationEnd;
    struct _populateVPMapLoopRepeat {
        inline void operator()(const CppDijkstraVisitor::VertexIterator& itr, CppDijkstraVisitor::VertexPredecessorMap& state, const CppDijkstraVisitor::VertexDescriptor& ctx) {
            return __for_iterator_loop.forLoopRepeat(itr, state, ctx);
        };
    };

    static CppDijkstraVisitor::_populateVPMapLoopRepeat populateVPMapLoopRepeat;
    struct _populateVPMapLoopStep {
        inline void operator()(const CppDijkstraVisitor::VertexIterator& itr, CppDijkstraVisitor::VertexPredecessorMap& vpm, const CppDijkstraVisitor::VertexDescriptor& vd) {
            CppDijkstraVisitor::VertexDescriptor v = CppDijkstraVisitor::vertexIterUnpack(itr);
            CppDijkstraVisitor::put(vpm, v, v);
        };
    };

private:
    static for_iterator_loop<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_populateVPMapLoopStep> __for_iterator_loop;
public:
    static CppDijkstraVisitor::_populateVPMapLoopStep populateVPMapLoopStep;
private:
    static read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> __read_write_property_map1;
    static two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppDijkstraVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex> __incidence_and_vertex_list_graph;
public:
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::Graph Graph;
    struct _numVertices {
        inline CppDijkstraVisitor::VertexCount operator()(const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.numVertices(g);
        };
    };

    static CppDijkstraVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppDijkstraVisitor::VertexCount operator()(const CppDijkstraVisitor::VertexDescriptor& v, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.outDegree(v, g);
        };
    };

    static CppDijkstraVisitor::_outDegree outDegree;
    struct _toVertexDescriptor {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::Vertex& v, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppDijkstraVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_graph.vertices(g, itr);
        };
    };

    static CppDijkstraVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::EdgeIterator EdgeIterator;
    typedef pair<CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::EdgeIterator>::Pair EdgeIteratorRange;
private:
    static pair<CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::EdgeIterator> __pair;
public:
    struct _edgeIterEnd {
        inline bool operator()(const CppDijkstraVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterEnd(ei);
        };
    };

    static CppDijkstraVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppDijkstraVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterNext(ei);
        };
    };

    static CppDijkstraVisitor::_edgeIterNext edgeIterNext;
    struct _iterRangeBegin {
        inline CppDijkstraVisitor::EdgeIterator operator()(const CppDijkstraVisitor::EdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppDijkstraVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppDijkstraVisitor::EdgeIterator operator()(const CppDijkstraVisitor::EdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppDijkstraVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeEdgeIteratorRange {
        inline CppDijkstraVisitor::EdgeIteratorRange operator()(const CppDijkstraVisitor::EdgeIterator& a, const CppDijkstraVisitor::EdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppDijkstraVisitor::_makeEdgeIteratorRange makeEdgeIteratorRange;
    struct _outEdges {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& v, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_graph.outEdges(v, g, itr);
        };
    };

    static CppDijkstraVisitor::_outEdges outEdges;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _edgeIterUnpack {
        inline CppDijkstraVisitor::EdgeDescriptor operator()(const CppDijkstraVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppDijkstraVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _src {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.src(e, g);
        };
    };

    static CppDijkstraVisitor::_src src;
    struct _tgt {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.tgt(e, g);
        };
    };

    static CppDijkstraVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppDijkstraVisitor::EdgeDescriptor operator()(const CppDijkstraVisitor::VertexDescriptor& v1, const CppDijkstraVisitor::VertexDescriptor& v2, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppDijkstraVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppDijkstraVisitor::Edge operator()(const CppDijkstraVisitor::Vertex& s, const CppDijkstraVisitor::Vertex& t) {
            return __incidence_and_vertex_list_graph.makeEdge(s, t);
        };
    };

    static CppDijkstraVisitor::_makeEdge makeEdge;
    typedef base_float_ops::Float Cost;
    typedef read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_edgeIterEnd, CppDijkstraVisitor::_edgeIterNext, CppDijkstraVisitor::_edgeIterUnpack>::PropertyMap EdgeCostMap;
    struct _emptyECMap {
        inline CppDijkstraVisitor::EdgeCostMap operator()() {
            return __read_write_property_map.emptyMap();
        };
    };

    static CppDijkstraVisitor::_emptyECMap emptyECMap;
    typedef read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::PropertyMap VertexCostMap;
    typedef triplet<CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::EdgeCostMap>::Triplet StateWithMaps;
    struct _getEdgeCostMap {
        inline CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::StateWithMaps& p) {
            return __triplet.third(p);
        };
    };

    static CppDijkstraVisitor::_getEdgeCostMap getEdgeCostMap;
    struct _getVertexPredecessorMap {
        inline CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::StateWithMaps& p) {
            return __triplet.second(p);
        };
    };

    static CppDijkstraVisitor::_getVertexPredecessorMap getVertexPredecessorMap;
    struct _putVertexPredecessorMap {
        inline CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::VertexPredecessorMap& vpm, const CppDijkstraVisitor::StateWithMaps& swm) {
            return CppDijkstraVisitor::makeStateWithMaps(CppDijkstraVisitor::getVertexCostMap(swm), vpm, CppDijkstraVisitor::getEdgeCostMap(swm));
        };
    };

    static CppDijkstraVisitor::_putVertexPredecessorMap putVertexPredecessorMap;
private:
    static triplet<CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::EdgeCostMap> __triplet;
public:
    struct _emptyVCMap {
        inline CppDijkstraVisitor::VertexCostMap operator()() {
            return __read_write_property_map0.emptyMap();
        };
    };

    static CppDijkstraVisitor::_emptyVCMap emptyVCMap;
    struct _getVertexCostMap {
        inline CppDijkstraVisitor::VertexCostMap operator()(const CppDijkstraVisitor::StateWithMaps& p) {
            return __triplet.first(p);
        };
    };

    static CppDijkstraVisitor::_getVertexCostMap getVertexCostMap;
    struct _makeStateWithMaps {
        inline CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::VertexCostMap& a, const CppDijkstraVisitor::VertexPredecessorMap& b, const CppDijkstraVisitor::EdgeCostMap& c) {
            return __triplet.makeTriplet(a, b, c);
        };
    };

    static CppDijkstraVisitor::_makeStateWithMaps makeStateWithMaps;
    struct _putVertexCostMap {
        inline CppDijkstraVisitor::StateWithMaps operator()(const CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::StateWithMaps& swm) {
            return CppDijkstraVisitor::makeStateWithMaps(vcm, CppDijkstraVisitor::getVertexPredecessorMap(swm), CppDijkstraVisitor::getEdgeCostMap(swm));
        };
    };

    static CppDijkstraVisitor::_putVertexCostMap putVertexCostMap;
    struct _relax {
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::EdgeCostMap& ecm, CppDijkstraVisitor::VertexCostMap& vcm, CppDijkstraVisitor::VertexPredecessorMap& vpm) {
            CppDijkstraVisitor::VertexDescriptor u = CppDijkstraVisitor::src(e, g);
            CppDijkstraVisitor::VertexDescriptor v = CppDijkstraVisitor::tgt(e, g);
            CppDijkstraVisitor::Cost uCost = CppDijkstraVisitor::get(vcm, u);
            CppDijkstraVisitor::Cost vCost = CppDijkstraVisitor::get(vcm, v);
            CppDijkstraVisitor::Cost edgeCost = CppDijkstraVisitor::get(ecm, e);
            if (CppDijkstraVisitor::less(CppDijkstraVisitor::plus(uCost, edgeCost), vCost))
            {
                CppDijkstraVisitor::put(vcm, v, CppDijkstraVisitor::plus(uCost, edgeCost));
                CppDijkstraVisitor::put(vpm, v, u);
            }
            else
                ;
        };
    };

    static CppDijkstraVisitor::_relax relax;
private:
    static read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_edgeIterEnd, CppDijkstraVisitor::_edgeIterNext, CppDijkstraVisitor::_edgeIterUnpack> __read_write_property_map;
    static read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> __read_write_property_map0;
public:
    struct _dijkstraShortestPaths {
        inline void operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::VertexDescriptor& start, CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::EdgeCostMap& ecm, const CppDijkstraVisitor::Cost& initialCost, CppDijkstraVisitor::VertexPredecessorMap& vpm) {
            CppDijkstraVisitor::put(vcm, start, initialCost);
            CppDijkstraVisitor::VertexIterator vertexItr;
            CppDijkstraVisitor::vertices(g, vertexItr);
            vpm = CppDijkstraVisitor::emptyVPMap();
            CppDijkstraVisitor::populateVPMapLoopRepeat(vertexItr, vpm, start);
            CppDijkstraVisitor::PriorityQueue pq = CppDijkstraVisitor::emptyPriorityQueue(vcm);
            CppDijkstraVisitor::StateWithMaps swm = CppDijkstraVisitor::makeStateWithMaps(vcm, vpm, ecm);
            CppDijkstraVisitor::ColorPropertyMap c = CppDijkstraVisitor::initMap(vertexItr, CppDijkstraVisitor::white());
            CppDijkstraVisitor::breadthFirstVisit(g, start, swm, pq, c);
            vcm = CppDijkstraVisitor::getVertexCostMap(swm);
            vpm = CppDijkstraVisitor::getVertexPredecessorMap(swm);
        };
    };

    static CppDijkstraVisitor::_dijkstraShortestPaths dijkstraShortestPaths;
    struct _less {
        inline bool operator()(const CppDijkstraVisitor::Cost& i1, const CppDijkstraVisitor::Cost& i2) {
            return __base_float_ops.less(i1, i2);
        };
    };

    static CppDijkstraVisitor::_less less;
    struct _plus {
        inline CppDijkstraVisitor::Cost operator()(const CppDijkstraVisitor::Cost& i1, const CppDijkstraVisitor::Cost& i2) {
            return __base_float_ops.plus(i1, i2);
        };
    };

    static CppDijkstraVisitor::_plus plus;
    typedef two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    typedef two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::Color Color;
    struct _black {
        inline CppDijkstraVisitor::Color operator()() {
            return __two_bit_color_map.black();
        };
    };

    static CppDijkstraVisitor::_black black;
    struct _get {
        inline CppDijkstraVisitor::Cost operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::VertexDescriptor& k) {
            return __read_write_property_map0.get(pm, k);
        };
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::VertexPredecessorMap& pm, const CppDijkstraVisitor::VertexDescriptor& k) {
            return __read_write_property_map1.get(pm, k);
        };
        inline CppDijkstraVisitor::Cost operator()(const CppDijkstraVisitor::EdgeCostMap& pm, const CppDijkstraVisitor::EdgeDescriptor& k) {
            return __read_write_property_map.get(pm, k);
        };
        inline CppDijkstraVisitor::Color operator()(const CppDijkstraVisitor::ColorPropertyMap& pm, const CppDijkstraVisitor::VertexDescriptor& k) {
            return __two_bit_color_map.get(pm, k);
        };
    };

    typedef priority_queue<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::Cost, CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::_get>::PriorityQueue PriorityQueue;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppDijkstraVisitor::EdgeIterator& itr, CppDijkstraVisitor::StateWithMaps& s1, CppDijkstraVisitor::PriorityQueue& s2, CppDijkstraVisitor::ColorPropertyMap& s3, const CppDijkstraVisitor::Graph& ctx1, const CppDijkstraVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppDijkstraVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppDijkstraVisitor::EdgeIterator& edgeItr, CppDijkstraVisitor::StateWithMaps& x, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c, const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::VertexDescriptor& u) {
            CppDijkstraVisitor::EdgeDescriptor e = CppDijkstraVisitor::edgeIterUnpack(edgeItr);
            CppDijkstraVisitor::VertexDescriptor v = CppDijkstraVisitor::tgt(e, g);
            CppDijkstraVisitor::examineEdge(e, g, q, x);
            CppDijkstraVisitor::Color vc = CppDijkstraVisitor::get(c, v);
            if ((vc) == (CppDijkstraVisitor::white()))
            {
                CppDijkstraVisitor::treeEdge(e, g, q, x);
                CppDijkstraVisitor::put(c, v, CppDijkstraVisitor::gray());
                CppDijkstraVisitor::discoverVertex(v, g, q, x);
                CppDijkstraVisitor::push(v, q);
            }
            else
                if ((vc) == (CppDijkstraVisitor::gray()))
                {
                    CppDijkstraVisitor::grayTarget(e, g, q, x);
                }
                else
                {
                    CppDijkstraVisitor::blackTarget(e, g, q, x);
                }
        };
    };

private:
    static for_iterator_loop3_2<CppDijkstraVisitor::Graph, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap, CppDijkstraVisitor::_edgeIterEnd, CppDijkstraVisitor::_edgeIterNext, CppDijkstraVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
public:
    static CppDijkstraVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
    struct _bfsOuterLoopCond {
        inline bool operator()(const CppDijkstraVisitor::StateWithMaps& a, const CppDijkstraVisitor::PriorityQueue& q, const CppDijkstraVisitor::ColorPropertyMap& c, const CppDijkstraVisitor::Graph& g) {
            return !CppDijkstraVisitor::isEmptyQueue(q);
        };
    };

    static CppDijkstraVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        inline void operator()(CppDijkstraVisitor::StateWithMaps& s1, CppDijkstraVisitor::PriorityQueue& s2, CppDijkstraVisitor::ColorPropertyMap& s3, const CppDijkstraVisitor::Graph& ctx) {
            return __while_loop3.repeat(s1, s2, s3, ctx);
        };
    };

    static CppDijkstraVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        inline void operator()(CppDijkstraVisitor::StateWithMaps& x, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c, const CppDijkstraVisitor::Graph& g) {
            CppDijkstraVisitor::VertexDescriptor u = CppDijkstraVisitor::front(q);
            CppDijkstraVisitor::pop(q);
            CppDijkstraVisitor::examineVertex(u, g, q, x);
            CppDijkstraVisitor::EdgeIterator edgeItr;
            CppDijkstraVisitor::outEdges(u, g, edgeItr);
            CppDijkstraVisitor::bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);
            CppDijkstraVisitor::put(c, u, CppDijkstraVisitor::black());
            CppDijkstraVisitor::finishVertex(u, g, q, x);
        };
    };

private:
    static while_loop3<CppDijkstraVisitor::Graph, CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap, CppDijkstraVisitor::_bfsOuterLoopCond, CppDijkstraVisitor::_bfsOuterLoopStep> __while_loop3;
public:
    static CppDijkstraVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _blackTarget {
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppDijkstraVisitor::_blackTarget blackTarget;
    struct _breadthFirstVisit {
        inline void operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::VertexDescriptor& s, CppDijkstraVisitor::StateWithMaps& a, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c) {
            CppDijkstraVisitor::discoverVertex(s, g, q, a);
            CppDijkstraVisitor::push(s, q);
            CppDijkstraVisitor::put(c, s, CppDijkstraVisitor::gray());
            CppDijkstraVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppDijkstraVisitor::_breadthFirstVisit breadthFirstVisit;
    struct _discoverVertex {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppDijkstraVisitor::_discoverVertex discoverVertex;
    struct _emptyPriorityQueue {
        inline CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::VertexCostMap& pm) {
            return __priority_queue.empty(pm);
        };
    };

    static CppDijkstraVisitor::_emptyPriorityQueue emptyPriorityQueue;
    struct _examineEdge {
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppDijkstraVisitor::_examineEdge examineEdge;
    struct _examineVertex {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppDijkstraVisitor::_examineVertex examineVertex;
    struct _finishVertex {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppDijkstraVisitor::_finishVertex finishVertex;
    struct _front {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::PriorityQueue& q) {
            return __priority_queue.front(q);
        };
    };

    static CppDijkstraVisitor::_front front;
    struct _grayTarget {
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& pq, CppDijkstraVisitor::StateWithMaps& swm) {
            CppDijkstraVisitor::VertexCostMap origVcm = CppDijkstraVisitor::getVertexCostMap(swm);
            CppDijkstraVisitor::VertexPredecessorMap vpm = CppDijkstraVisitor::getVertexPredecessorMap(swm);
            CppDijkstraVisitor::EdgeCostMap ecm = CppDijkstraVisitor::getEdgeCostMap(swm);
            CppDijkstraVisitor::VertexCostMap vcm = origVcm;
            CppDijkstraVisitor::relax(e, g, ecm, vcm, vpm);
            if ((vcm) == (origVcm))
                ;
            else
            {
                swm = CppDijkstraVisitor::putVertexPredecessorMap(vpm, CppDijkstraVisitor::putVertexCostMap(vcm, swm));
                pq = CppDijkstraVisitor::update(vcm, CppDijkstraVisitor::tgt(e, g), pq);
            }
        };
    };

    static CppDijkstraVisitor::_grayTarget grayTarget;
    struct _isEmptyQueue {
        inline bool operator()(const CppDijkstraVisitor::PriorityQueue& q) {
            return __priority_queue.isEmpty(q);
        };
    };

    static CppDijkstraVisitor::_isEmptyQueue isEmptyQueue;
    struct _nonTreeEdge {
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppDijkstraVisitor::_nonTreeEdge nonTreeEdge;
    struct _pop {
        inline void operator()(CppDijkstraVisitor::PriorityQueue& q) {
            return __priority_queue.pop(q);
        };
    };

    static CppDijkstraVisitor::_pop pop;
    struct _push {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& a, CppDijkstraVisitor::PriorityQueue& q) {
            return __priority_queue.push(a, q);
        };
    };

    static CppDijkstraVisitor::_push push;
    struct _treeEdge {
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& pq, CppDijkstraVisitor::StateWithMaps& swm) {
            CppDijkstraVisitor::VertexCostMap vcm = CppDijkstraVisitor::getVertexCostMap(swm);
            CppDijkstraVisitor::VertexPredecessorMap vpm = CppDijkstraVisitor::getVertexPredecessorMap(swm);
            CppDijkstraVisitor::EdgeCostMap ecm = CppDijkstraVisitor::getEdgeCostMap(swm);
            CppDijkstraVisitor::relax(e, g, ecm, vcm, vpm);
            swm = CppDijkstraVisitor::putVertexPredecessorMap(vpm, CppDijkstraVisitor::putVertexCostMap(vcm, swm));
        };
    };

    static CppDijkstraVisitor::_treeEdge treeEdge;
    struct _update {
        inline CppDijkstraVisitor::PriorityQueue operator()(const CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::VertexDescriptor& a, const CppDijkstraVisitor::PriorityQueue& pq) {
            return __priority_queue.update(pm, a, pq);
        };
    };

    static CppDijkstraVisitor::_update update;
private:
    static priority_queue<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::Cost, CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::_get> __priority_queue;
public:
    static CppDijkstraVisitor::_get get;
    struct _gray {
        inline CppDijkstraVisitor::Color operator()() {
            return __two_bit_color_map.gray();
        };
    };

    static CppDijkstraVisitor::_gray gray;
    struct _initMap {
        inline CppDijkstraVisitor::VertexCostMap operator()(const CppDijkstraVisitor::VertexIterator& kli, const CppDijkstraVisitor::Cost& v) {
            return __read_write_property_map0.initMap(kli, v);
        };
        inline CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::VertexIterator& kli, const CppDijkstraVisitor::VertexDescriptor& v) {
            return __read_write_property_map1.initMap(kli, v);
        };
        inline CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::EdgeIterator& kli, const CppDijkstraVisitor::Cost& v) {
            return __read_write_property_map.initMap(kli, v);
        };
        inline CppDijkstraVisitor::ColorPropertyMap operator()(const CppDijkstraVisitor::VertexIterator& kli, const CppDijkstraVisitor::Color& v) {
            return __two_bit_color_map.initMap(kli, v);
        };
    };

    static CppDijkstraVisitor::_initMap initMap;
    struct _put {
        inline void operator()(CppDijkstraVisitor::VertexCostMap& pm, const CppDijkstraVisitor::VertexDescriptor& k, const CppDijkstraVisitor::Cost& v) {
            return __read_write_property_map0.put(pm, k, v);
        };
        inline void operator()(CppDijkstraVisitor::VertexPredecessorMap& pm, const CppDijkstraVisitor::VertexDescriptor& k, const CppDijkstraVisitor::VertexDescriptor& v) {
            return __read_write_property_map1.put(pm, k, v);
        };
        inline void operator()(CppDijkstraVisitor::EdgeCostMap& pm, const CppDijkstraVisitor::EdgeDescriptor& k, const CppDijkstraVisitor::Cost& v) {
            return __read_write_property_map.put(pm, k, v);
        };
        inline void operator()(CppDijkstraVisitor::ColorPropertyMap& pm, const CppDijkstraVisitor::VertexDescriptor& k, const CppDijkstraVisitor::Color& v) {
            return __two_bit_color_map.put(pm, k, v);
        };
    };

    static CppDijkstraVisitor::_put put;
    struct _white {
        inline CppDijkstraVisitor::Color operator()() {
            return __two_bit_color_map.white();
        };
    };

    static CppDijkstraVisitor::_white white;
};
} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp

namespace examples {
namespace bgl_v2 {
namespace mg_src {
namespace bgl_v2_cpp {
struct CppParallelBFSTestVisitor {
private:
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::VertexDescriptor VertexDescriptor;
    typedef thread_safe_vector<CppParallelBFSTestVisitor::VertexDescriptor>::Vector VertexVector;
    struct _emptyVertexVector {
        inline CppParallelBFSTestVisitor::VertexVector operator()() {
            return __thread_safe_vector.empty();
        };
    };

    static CppParallelBFSTestVisitor::_emptyVertexVector emptyVertexVector;
private:
    static thread_safe_fifo_queue<CppParallelBFSTestVisitor::VertexDescriptor> __thread_safe_fifo_queue;
    static thread_safe_vector<CppParallelBFSTestVisitor::VertexDescriptor> __thread_safe_vector;
public:
    struct _pushBack {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& a, CppParallelBFSTestVisitor::VertexVector& v) {
            return __thread_safe_vector.pushBack(a, v);
        };
    };

    static CppParallelBFSTestVisitor::_pushBack pushBack;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppParallelBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterEnd(ei);
        };
    };

    static CppParallelBFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppParallelBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterNext(ei);
        };
    };

    static CppParallelBFSTestVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_graph.vertexIterUnpack(ei);
        };
    };

private:
    static two_bit_color_map<CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::VertexIterator, CppParallelBFSTestVisitor::_vertexIterEnd, CppParallelBFSTestVisitor::_vertexIterNext, CppParallelBFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppParallelBFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex> __incidence_and_vertex_list_graph;
public:
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::Graph Graph;
    struct _breadthFirstSearch {
        inline CppParallelBFSTestVisitor::VertexVector operator()(const CppParallelBFSTestVisitor::Graph& g, const CppParallelBFSTestVisitor::VertexDescriptor& start, const CppParallelBFSTestVisitor::VertexVector& init) {
            CppParallelBFSTestVisitor::FIFOQueue q = CppParallelBFSTestVisitor::empty();
            CppParallelBFSTestVisitor::VertexIterator vertexItr;
            CppParallelBFSTestVisitor::vertices(g, vertexItr);
            CppParallelBFSTestVisitor::ColorPropertyMap c = CppParallelBFSTestVisitor::initMap(vertexItr, CppParallelBFSTestVisitor::white());
            CppParallelBFSTestVisitor::VertexVector a = init;
            CppParallelBFSTestVisitor::breadthFirstVisit(g, start, a, q, c);
            return a;
        };
    };

    static CppParallelBFSTestVisitor::_breadthFirstSearch breadthFirstSearch;
    struct _numVertices {
        inline CppParallelBFSTestVisitor::VertexCount operator()(const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.numVertices(g);
        };
    };

    static CppParallelBFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppParallelBFSTestVisitor::VertexCount operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.outDegree(v, g);
        };
    };

    static CppParallelBFSTestVisitor::_outDegree outDegree;
    struct _toVertexDescriptor {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::Vertex& v, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppParallelBFSTestVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_graph.vertices(g, itr);
        };
    };

    static CppParallelBFSTestVisitor::_vertices vertices;
    typedef thread_safe_fifo_queue<CppParallelBFSTestVisitor::VertexDescriptor>::FIFOQueue FIFOQueue;
    struct _empty {
        inline CppParallelBFSTestVisitor::FIFOQueue operator()() {
            return __thread_safe_fifo_queue.empty();
        };
    };

    static CppParallelBFSTestVisitor::_empty empty;
    struct _examineVertex {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v, const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::VertexVector& a) {
            CppParallelBFSTestVisitor::pushBack(v, a);
        };
    };

    static CppParallelBFSTestVisitor::_examineVertex examineVertex;
    struct _front {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::FIFOQueue& q) {
            return __thread_safe_fifo_queue.front(q);
        };
    };

    static CppParallelBFSTestVisitor::_front front;
    struct _isEmptyQueue {
        inline bool operator()(const CppParallelBFSTestVisitor::FIFOQueue& q) {
            return __thread_safe_fifo_queue.isEmpty(q);
        };
    };

    static CppParallelBFSTestVisitor::_isEmptyQueue isEmptyQueue;
    struct _pop {
        inline void operator()(CppParallelBFSTestVisitor::FIFOQueue& q) {
            return __thread_safe_fifo_queue.pop(q);
        };
    };

    static CppParallelBFSTestVisitor::_pop pop;
    struct _push {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& a, CppParallelBFSTestVisitor::FIFOQueue& q) {
            return __thread_safe_fifo_queue.push(a, q);
        };
    };

    static CppParallelBFSTestVisitor::_push push;
    struct _pushPopBehavior {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& a, const CppParallelBFSTestVisitor::FIFOQueue& inq) {
            CppParallelBFSTestVisitor::FIFOQueue mut_inq = inq;
            CppParallelBFSTestVisitor::push(a, mut_inq);
            assert((CppParallelBFSTestVisitor::front(mut_inq)) == (a));
            CppParallelBFSTestVisitor::pop(mut_inq);
            assert((inq) == (mut_inq));
        };
    };

    static CppParallelBFSTestVisitor::_pushPopBehavior pushPopBehavior;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::EdgeIterator EdgeIterator;
    typedef pair<CppParallelBFSTestVisitor::EdgeIterator, CppParallelBFSTestVisitor::EdgeIterator>::Pair EdgeIteratorRange;
private:
    static pair<CppParallelBFSTestVisitor::EdgeIterator, CppParallelBFSTestVisitor::EdgeIterator> __pair;
public:
    struct _edgeIterEnd {
        inline bool operator()(const CppParallelBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterEnd(ei);
        };
    };

    static CppParallelBFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppParallelBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterNext(ei);
        };
    };

    static CppParallelBFSTestVisitor::_edgeIterNext edgeIterNext;
    struct _iterRangeBegin {
        inline CppParallelBFSTestVisitor::EdgeIterator operator()(const CppParallelBFSTestVisitor::EdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppParallelBFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppParallelBFSTestVisitor::EdgeIterator operator()(const CppParallelBFSTestVisitor::EdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppParallelBFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeEdgeIteratorRange {
        inline CppParallelBFSTestVisitor::EdgeIteratorRange operator()(const CppParallelBFSTestVisitor::EdgeIterator& a, const CppParallelBFSTestVisitor::EdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppParallelBFSTestVisitor::_makeEdgeIteratorRange makeEdgeIteratorRange;
    struct _outEdges {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v, const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_graph.outEdges(v, g, itr);
        };
    };

    static CppParallelBFSTestVisitor::_outEdges outEdges;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _defaultAction {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& edgeOrVertex, const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::VertexVector& a) {
            ;
        };
        inline void operator()(const CppParallelBFSTestVisitor::EdgeDescriptor& edgeOrVertex, const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::VertexVector& a) {
            ;
        };
    };

    static CppParallelBFSTestVisitor::_defaultAction defaultAction;
    struct _edgeIterUnpack {
        inline CppParallelBFSTestVisitor::EdgeDescriptor operator()(const CppParallelBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppParallelBFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _src {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::EdgeDescriptor& e, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.src(e, g);
        };
    };

    static CppParallelBFSTestVisitor::_src src;
    struct _tgt {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::EdgeDescriptor& e, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.tgt(e, g);
        };
    };

    static CppParallelBFSTestVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppParallelBFSTestVisitor::EdgeDescriptor operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v1, const CppParallelBFSTestVisitor::VertexDescriptor& v2, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppParallelBFSTestVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_graph<CppParallelBFSTestVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppParallelBFSTestVisitor::Edge operator()(const CppParallelBFSTestVisitor::Vertex& s, const CppParallelBFSTestVisitor::Vertex& t) {
            return __incidence_and_vertex_list_graph.makeEdge(s, t);
        };
    };

    static CppParallelBFSTestVisitor::_makeEdge makeEdge;
    typedef two_bit_color_map<CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::VertexIterator, CppParallelBFSTestVisitor::_vertexIterEnd, CppParallelBFSTestVisitor::_vertexIterNext, CppParallelBFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppParallelBFSTestVisitor::EdgeIterator& itr, CppParallelBFSTestVisitor::VertexVector& s1, CppParallelBFSTestVisitor::FIFOQueue& s2, CppParallelBFSTestVisitor::ColorPropertyMap& s3, const CppParallelBFSTestVisitor::Graph& ctx1, const CppParallelBFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_parallel_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppParallelBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppParallelBFSTestVisitor::EdgeIterator& edgeItr, CppParallelBFSTestVisitor::VertexVector& x, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::ColorPropertyMap& c, const CppParallelBFSTestVisitor::Graph& g, const CppParallelBFSTestVisitor::VertexDescriptor& u) {
            CppParallelBFSTestVisitor::EdgeDescriptor e = CppParallelBFSTestVisitor::edgeIterUnpack(edgeItr);
            CppParallelBFSTestVisitor::VertexDescriptor v = CppParallelBFSTestVisitor::tgt(e, g);
            CppParallelBFSTestVisitor::defaultAction(e, g, q, x);
            CppParallelBFSTestVisitor::Color vc = CppParallelBFSTestVisitor::get(c, v);
            if ((vc) == (CppParallelBFSTestVisitor::white()))
            {
                CppParallelBFSTestVisitor::defaultAction(e, g, q, x);
                CppParallelBFSTestVisitor::put(c, v, CppParallelBFSTestVisitor::gray());
                CppParallelBFSTestVisitor::defaultAction(v, g, q, x);
                CppParallelBFSTestVisitor::push(v, q);
            }
            else
                if ((vc) == (CppParallelBFSTestVisitor::gray()))
                {
                    CppParallelBFSTestVisitor::defaultAction(e, g, q, x);
                }
                else
                {
                    CppParallelBFSTestVisitor::defaultAction(e, g, q, x);
                }
        };
    };

private:
    static for_parallel_iterator_loop3_2<CppParallelBFSTestVisitor::Graph, CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::EdgeIterator, CppParallelBFSTestVisitor::VertexVector, CppParallelBFSTestVisitor::FIFOQueue, CppParallelBFSTestVisitor::ColorPropertyMap, CppParallelBFSTestVisitor::_edgeIterEnd, CppParallelBFSTestVisitor::_edgeIterNext, CppParallelBFSTestVisitor::_bfsInnerLoopStep> __for_parallel_iterator_loop3_2;
public:
    static CppParallelBFSTestVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
    struct _bfsOuterLoopCond {
        inline bool operator()(const CppParallelBFSTestVisitor::VertexVector& a, const CppParallelBFSTestVisitor::FIFOQueue& q, const CppParallelBFSTestVisitor::ColorPropertyMap& c, const CppParallelBFSTestVisitor::Graph& g) {
            return !CppParallelBFSTestVisitor::isEmptyQueue(q);
        };
    };

    static CppParallelBFSTestVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        inline void operator()(CppParallelBFSTestVisitor::VertexVector& s1, CppParallelBFSTestVisitor::FIFOQueue& s2, CppParallelBFSTestVisitor::ColorPropertyMap& s3, const CppParallelBFSTestVisitor::Graph& ctx) {
            return __while_loop3.repeat(s1, s2, s3, ctx);
        };
    };

    static CppParallelBFSTestVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        inline void operator()(CppParallelBFSTestVisitor::VertexVector& x, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::ColorPropertyMap& c, const CppParallelBFSTestVisitor::Graph& g) {
            CppParallelBFSTestVisitor::VertexDescriptor u = CppParallelBFSTestVisitor::front(q);
            CppParallelBFSTestVisitor::pop(q);
            CppParallelBFSTestVisitor::examineVertex(u, g, q, x);
            CppParallelBFSTestVisitor::EdgeIterator edgeItr;
            CppParallelBFSTestVisitor::outEdges(u, g, edgeItr);
            CppParallelBFSTestVisitor::bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);
            CppParallelBFSTestVisitor::put(c, u, CppParallelBFSTestVisitor::black());
            CppParallelBFSTestVisitor::defaultAction(u, g, q, x);
        };
    };

private:
    static while_loop3<CppParallelBFSTestVisitor::Graph, CppParallelBFSTestVisitor::VertexVector, CppParallelBFSTestVisitor::FIFOQueue, CppParallelBFSTestVisitor::ColorPropertyMap, CppParallelBFSTestVisitor::_bfsOuterLoopCond, CppParallelBFSTestVisitor::_bfsOuterLoopStep> __while_loop3;
public:
    static CppParallelBFSTestVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _breadthFirstVisit {
        inline void operator()(const CppParallelBFSTestVisitor::Graph& g, const CppParallelBFSTestVisitor::VertexDescriptor& s, CppParallelBFSTestVisitor::VertexVector& a, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::ColorPropertyMap& c) {
            CppParallelBFSTestVisitor::defaultAction(s, g, q, a);
            CppParallelBFSTestVisitor::push(s, q);
            CppParallelBFSTestVisitor::put(c, s, CppParallelBFSTestVisitor::gray());
            CppParallelBFSTestVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppParallelBFSTestVisitor::_breadthFirstVisit breadthFirstVisit;
    typedef two_bit_color_map<CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::VertexIterator, CppParallelBFSTestVisitor::_vertexIterEnd, CppParallelBFSTestVisitor::_vertexIterNext, CppParallelBFSTestVisitor::_vertexIterUnpack>::Color Color;
    struct _black {
        inline CppParallelBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.black();
        };
    };

    static CppParallelBFSTestVisitor::_black black;
    struct _get {
        inline CppParallelBFSTestVisitor::Color operator()(const CppParallelBFSTestVisitor::ColorPropertyMap& pm, const CppParallelBFSTestVisitor::VertexDescriptor& k) {
            return __two_bit_color_map.get(pm, k);
        };
    };

    static CppParallelBFSTestVisitor::_get get;
    struct _gray {
        inline CppParallelBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.gray();
        };
    };

    static CppParallelBFSTestVisitor::_gray gray;
    struct _initMap {
        inline CppParallelBFSTestVisitor::ColorPropertyMap operator()(const CppParallelBFSTestVisitor::VertexIterator& kli, const CppParallelBFSTestVisitor::Color& v) {
            return __two_bit_color_map.initMap(kli, v);
        };
    };

    static CppParallelBFSTestVisitor::_initMap initMap;
    struct _put {
        inline void operator()(CppParallelBFSTestVisitor::ColorPropertyMap& pm, const CppParallelBFSTestVisitor::VertexDescriptor& k, const CppParallelBFSTestVisitor::Color& v) {
            return __two_bit_color_map.put(pm, k, v);
        };
    };

    static CppParallelBFSTestVisitor::_put put;
    struct _white {
        inline CppParallelBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.white();
        };
    };

    static CppParallelBFSTestVisitor::_white white;
};
} // examples
} // bgl_v2
} // mg_src
} // bgl_v2_cpp