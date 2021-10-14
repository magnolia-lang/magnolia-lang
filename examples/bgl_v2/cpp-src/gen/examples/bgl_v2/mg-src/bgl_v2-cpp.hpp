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
    static two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
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
            CppBFSTestVisitor::VertexIterator vertexBeg;
            CppBFSTestVisitor::VertexIterator vertexEnd;
            CppBFSTestVisitor::vertices(g, vertexBeg, vertexEnd);
            CppBFSTestVisitor::ColorPropertyMap c = CppBFSTestVisitor::initMap(vertexBeg, vertexEnd, CppBFSTestVisitor::white());
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
        inline void operator()(const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::VertexIterator& itBeg, CppBFSTestVisitor::VertexIterator& itEnd) {
            return __incidence_and_vertex_list_graph.vertices(g, itBeg, itEnd);
        };
    };

    static CppBFSTestVisitor::_vertices vertices;
    typedef fifo_queue<CppBFSTestVisitor::VertexDescriptor>::FIFOQueue FIFOQueue;
    struct _discoverVertex {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::VertexVector& a) {
            CppBFSTestVisitor::pushBack(v, a);
        };
    };

    static CppBFSTestVisitor::_discoverVertex discoverVertex;
    struct _empty {
        inline CppBFSTestVisitor::FIFOQueue operator()() {
            return __fifo_queue.empty();
        };
    };

    static CppBFSTestVisitor::_empty empty;
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
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::EdgeIterator& itBeg, CppBFSTestVisitor::EdgeIterator& itEnd) {
            return __incidence_and_vertex_list_graph.outEdges(v, g, itBeg, itEnd);
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
    typedef two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppBFSTestVisitor::EdgeIterator& itr, const CppBFSTestVisitor::EdgeIterator& endItr, CppBFSTestVisitor::VertexVector& s1, CppBFSTestVisitor::FIFOQueue& s2, CppBFSTestVisitor::ColorPropertyMap& s3, const CppBFSTestVisitor::Graph& ctx1, const CppBFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, endItr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppBFSTestVisitor::EdgeIterator& edgeItr, const CppBFSTestVisitor::EdgeIterator& edgeItrEnd, CppBFSTestVisitor::VertexVector& x, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c, const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::VertexDescriptor& u) {
            CppBFSTestVisitor::EdgeDescriptor e = CppBFSTestVisitor::edgeIterUnpack(edgeItr);
            CppBFSTestVisitor::VertexDescriptor v = CppBFSTestVisitor::tgt(e, g);
            CppBFSTestVisitor::defaultAction(e, g, q, x);
            CppBFSTestVisitor::Color vc = CppBFSTestVisitor::get(c, v);
            if ((vc) == (CppBFSTestVisitor::white()))
            {
                CppBFSTestVisitor::defaultAction(e, g, q, x);
                CppBFSTestVisitor::put(c, v, CppBFSTestVisitor::gray());
                CppBFSTestVisitor::discoverVertex(v, g, q, x);
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
    static for_iterator_loop3_2<CppBFSTestVisitor::Graph, CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::EdgeIterator, CppBFSTestVisitor::VertexVector, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap, CppBFSTestVisitor::_edgeIterNext, CppBFSTestVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
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
            CppBFSTestVisitor::defaultAction(u, g, q, x);
            CppBFSTestVisitor::EdgeIterator edgeItrBegin;
            CppBFSTestVisitor::EdgeIterator edgeItrEnd;
            CppBFSTestVisitor::outEdges(u, g, edgeItrBegin, edgeItrEnd);
            CppBFSTestVisitor::bfsInnerLoopRepeat(edgeItrBegin, edgeItrEnd, x, q, c, g, u);
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
            CppBFSTestVisitor::discoverVertex(s, g, q, a);
            CppBFSTestVisitor::push(s, q);
            CppBFSTestVisitor::put(c, s, CppBFSTestVisitor::gray());
            CppBFSTestVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppBFSTestVisitor::_breadthFirstVisit breadthFirstVisit;
    typedef two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack>::Color Color;
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
        inline CppBFSTestVisitor::ColorPropertyMap operator()(const CppBFSTestVisitor::VertexIterator& klBeg, const CppBFSTestVisitor::VertexIterator& klEnd, const CppBFSTestVisitor::Color& v) {
            return __two_bit_color_map.initMap(klBeg, klEnd, v);
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

    typedef read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::PropertyMap VertexPredecessorMap;
    struct _emptyVPMap {
        inline CppDijkstraVisitor::VertexPredecessorMap operator()() {
            return __read_write_property_map1.emptyMap();
        };
    };

    static CppDijkstraVisitor::_emptyVPMap emptyVPMap;
    struct _forIterationEnd {
        inline void operator()(const CppDijkstraVisitor::VertexIterator& itr, const CppDijkstraVisitor::VertexIterator& endItr, const CppDijkstraVisitor::VertexPredecessorMap& state, const CppDijkstraVisitor::VertexDescriptor& ctx) {
            CppDijkstraVisitor::VertexPredecessorMap mut_state = state;
            if ((itr) == (endItr))
            {
                CppDijkstraVisitor::populateVPMapLoopRepeat(itr, endItr, mut_state, ctx);
                assert((mut_state) == (state));
            }
            else
                ;
        };
    };

    static CppDijkstraVisitor::_forIterationEnd forIterationEnd;
    struct _populateVPMapLoopRepeat {
        inline void operator()(const CppDijkstraVisitor::VertexIterator& itr, const CppDijkstraVisitor::VertexIterator& endItr, CppDijkstraVisitor::VertexPredecessorMap& state, const CppDijkstraVisitor::VertexDescriptor& ctx) {
            return __for_iterator_loop.forLoopRepeat(itr, endItr, state, ctx);
        };
    };

    static CppDijkstraVisitor::_populateVPMapLoopRepeat populateVPMapLoopRepeat;
    struct _populateVPMapLoopStep {
        inline void operator()(const CppDijkstraVisitor::VertexIterator& itr, const CppDijkstraVisitor::VertexIterator& endItr, CppDijkstraVisitor::VertexPredecessorMap& vpm, const CppDijkstraVisitor::VertexDescriptor& vd) {
            CppDijkstraVisitor::VertexDescriptor v = CppDijkstraVisitor::vertexIterUnpack(itr);
            CppDijkstraVisitor::put(vpm, v, v);
        };
    };

private:
    static for_iterator_loop<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_populateVPMapLoopStep> __for_iterator_loop;
public:
    static CppDijkstraVisitor::_populateVPMapLoopStep populateVPMapLoopStep;
private:
    static read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> __read_write_property_map1;
    static two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> __two_bit_color_map;
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
        inline void operator()(const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::VertexIterator& itBeg, CppDijkstraVisitor::VertexIterator& itEnd) {
            return __incidence_and_vertex_list_graph.vertices(g, itBeg, itEnd);
        };
    };

    static CppDijkstraVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_graph<CppDijkstraVisitor::Vertex>::EdgeIterator EdgeIterator;
    typedef pair<CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::EdgeIterator>::Pair EdgeIteratorRange;
private:
    static pair<CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::EdgeIterator> __pair;
public:
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
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& v, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::EdgeIterator& itBeg, CppDijkstraVisitor::EdgeIterator& itEnd) {
            return __incidence_and_vertex_list_graph.outEdges(v, g, itBeg, itEnd);
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
    typedef read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_edgeIterNext, CppDijkstraVisitor::_edgeIterUnpack>::PropertyMap EdgeCostMap;
    struct _emptyECMap {
        inline CppDijkstraVisitor::EdgeCostMap operator()() {
            return __read_write_property_map.emptyMap();
        };
    };

    static CppDijkstraVisitor::_emptyECMap emptyECMap;
    typedef read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::PropertyMap VertexCostMap;
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
    static read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_edgeIterNext, CppDijkstraVisitor::_edgeIterUnpack> __read_write_property_map;
    static read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> __read_write_property_map0;
public:
    struct _dijkstraShortestPaths {
        inline void operator()(const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::VertexDescriptor& start, CppDijkstraVisitor::VertexCostMap& vcm, const CppDijkstraVisitor::EdgeCostMap& ecm, const CppDijkstraVisitor::Cost& initialCost, CppDijkstraVisitor::VertexPredecessorMap& vpm) {
            CppDijkstraVisitor::put(vcm, start, initialCost);
            CppDijkstraVisitor::VertexIterator vertexBeg;
            CppDijkstraVisitor::VertexIterator vertexEnd;
            CppDijkstraVisitor::vertices(g, vertexBeg, vertexEnd);
            vpm = CppDijkstraVisitor::emptyVPMap();
            CppDijkstraVisitor::populateVPMapLoopRepeat(vertexBeg, vertexEnd, vpm, start);
            CppDijkstraVisitor::PriorityQueue pq = CppDijkstraVisitor::emptyPriorityQueue(vcm);
            CppDijkstraVisitor::StateWithMaps swm = CppDijkstraVisitor::makeStateWithMaps(vcm, vpm, ecm);
            CppDijkstraVisitor::ColorPropertyMap c = CppDijkstraVisitor::initMap(vertexBeg, vertexEnd, CppDijkstraVisitor::white());
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
    typedef two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    typedef two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack>::Color Color;
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
        inline void operator()(const CppDijkstraVisitor::EdgeIterator& itr, const CppDijkstraVisitor::EdgeIterator& endItr, CppDijkstraVisitor::StateWithMaps& s1, CppDijkstraVisitor::PriorityQueue& s2, CppDijkstraVisitor::ColorPropertyMap& s3, const CppDijkstraVisitor::Graph& ctx1, const CppDijkstraVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, endItr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppDijkstraVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppDijkstraVisitor::EdgeIterator& edgeItr, const CppDijkstraVisitor::EdgeIterator& edgeItrEnd, CppDijkstraVisitor::StateWithMaps& x, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c, const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::VertexDescriptor& u) {
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
    static for_iterator_loop3_2<CppDijkstraVisitor::Graph, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::EdgeIterator, CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap, CppDijkstraVisitor::_edgeIterNext, CppDijkstraVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
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
            CppDijkstraVisitor::EdgeIterator edgeItrBegin;
            CppDijkstraVisitor::EdgeIterator edgeItrEnd;
            CppDijkstraVisitor::outEdges(u, g, edgeItrBegin, edgeItrEnd);
            CppDijkstraVisitor::bfsInnerLoopRepeat(edgeItrBegin, edgeItrEnd, x, q, c, g, u);
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
        inline CppDijkstraVisitor::VertexCostMap operator()(const CppDijkstraVisitor::VertexIterator& klBeg, const CppDijkstraVisitor::VertexIterator& klEnd, const CppDijkstraVisitor::Cost& v) {
            return __read_write_property_map0.initMap(klBeg, klEnd, v);
        };
        inline CppDijkstraVisitor::VertexPredecessorMap operator()(const CppDijkstraVisitor::VertexIterator& klBeg, const CppDijkstraVisitor::VertexIterator& klEnd, const CppDijkstraVisitor::VertexDescriptor& v) {
            return __read_write_property_map1.initMap(klBeg, klEnd, v);
        };
        inline CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::EdgeIterator& klBeg, const CppDijkstraVisitor::EdgeIterator& klEnd, const CppDijkstraVisitor::Cost& v) {
            return __read_write_property_map.initMap(klBeg, klEnd, v);
        };
        inline CppDijkstraVisitor::ColorPropertyMap operator()(const CppDijkstraVisitor::VertexIterator& klBeg, const CppDijkstraVisitor::VertexIterator& klEnd, const CppDijkstraVisitor::Color& v) {
            return __two_bit_color_map.initMap(klBeg, klEnd, v);
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