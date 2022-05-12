#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
struct CppBFSTestVisitor {
private:
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::VertexDescriptor VertexDescriptor;
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterEnd(ei);
        };
    };

    static CppBFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterNext(ei);
        };
    };

    static CppBFSTestVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterUnpack(ei);
        };
    };

private:
    static two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterEnd, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppBFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex> __incidence_and_vertex_list_and_edge_list_graph;
public:
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::OutEdgeIterator OutEdgeIterator;
    typedef pair<CppBFSTestVisitor::OutEdgeIterator, CppBFSTestVisitor::OutEdgeIterator>::Pair OutEdgeIteratorRange;
private:
    static pair<CppBFSTestVisitor::OutEdgeIterator, CppBFSTestVisitor::OutEdgeIterator> __pair;
public:
    struct _iterRangeBegin {
        inline CppBFSTestVisitor::OutEdgeIterator operator()(const CppBFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppBFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppBFSTestVisitor::OutEdgeIterator operator()(const CppBFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppBFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeOutEdgeIteratorRange {
        inline CppBFSTestVisitor::OutEdgeIteratorRange operator()(const CppBFSTestVisitor::OutEdgeIterator& a, const CppBFSTestVisitor::OutEdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppBFSTestVisitor::_makeOutEdgeIteratorRange makeOutEdgeIteratorRange;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppBFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterEnd(ei);
        };
    };

    static CppBFSTestVisitor::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppBFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterNext(ei);
        };
    };

    static CppBFSTestVisitor::_outEdgeIterNext outEdgeIterNext;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::Graph Graph;
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
            return __incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppBFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppBFSTestVisitor::VertexCount operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppBFSTestVisitor::_outDegree outDegree;
    struct _outEdges {
        inline void operator()(const CppBFSTestVisitor::VertexDescriptor& v, const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::OutEdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppBFSTestVisitor::_outEdges outEdges;
    struct _toVertexDescriptor {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::Vertex& v, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppBFSTestVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::EdgeIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterEnd(ei);
        };
    };

    static CppBFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterNext(ei);
        };
    };

    static CppBFSTestVisitor::_edgeIterNext edgeIterNext;
    struct _edges {
        inline void operator()(const CppBFSTestVisitor::Graph& g, CppBFSTestVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppBFSTestVisitor::_edges edges;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
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
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppBFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _outEdgeIterUnpack {
        inline CppBFSTestVisitor::EdgeDescriptor operator()(const CppBFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterUnpack(ei);
        };
    };

    static CppBFSTestVisitor::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _src {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::EdgeDescriptor& e, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppBFSTestVisitor::_src src;
    struct _tgt {
        inline CppBFSTestVisitor::VertexDescriptor operator()(const CppBFSTestVisitor::EdgeDescriptor& e, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppBFSTestVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppBFSTestVisitor::EdgeDescriptor operator()(const CppBFSTestVisitor::VertexDescriptor& v1, const CppBFSTestVisitor::VertexDescriptor& v2, const CppBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppBFSTestVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppBFSTestVisitor::Edge operator()(const CppBFSTestVisitor::Vertex& s, const CppBFSTestVisitor::Vertex& t) {
            return __incidence_and_vertex_list_and_edge_list_graph.makeEdge(s, t);
        };
    };

    static CppBFSTestVisitor::_makeEdge makeEdge;
    typedef two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterEnd, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppBFSTestVisitor::OutEdgeIterator& itr, CppBFSTestVisitor::VertexVector& s1, CppBFSTestVisitor::FIFOQueue& s2, CppBFSTestVisitor::ColorPropertyMap& s3, const CppBFSTestVisitor::Graph& ctx1, const CppBFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppBFSTestVisitor::OutEdgeIterator& edgeItr, CppBFSTestVisitor::VertexVector& x, CppBFSTestVisitor::FIFOQueue& q, CppBFSTestVisitor::ColorPropertyMap& c, const CppBFSTestVisitor::Graph& g, const CppBFSTestVisitor::VertexDescriptor& u) {
            CppBFSTestVisitor::EdgeDescriptor e = CppBFSTestVisitor::outEdgeIterUnpack(edgeItr);
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
    static for_iterator_loop3_2<CppBFSTestVisitor::Graph, CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::OutEdgeIterator, CppBFSTestVisitor::VertexVector, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap, CppBFSTestVisitor::_outEdgeIterEnd, CppBFSTestVisitor::_outEdgeIterNext, CppBFSTestVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
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
            CppBFSTestVisitor::OutEdgeIterator edgeItr;
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
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
struct CppBellmanFord {
private:
    static base_unit __base_unit;
    static base_types __base_types;
    static base_float_ops __base_float_ops;
    static base_bool __base_bool;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::VertexDescriptor VertexDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppBellmanFord::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterEnd(ei);
        };
    };

    static CppBellmanFord::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppBellmanFord::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterNext(ei);
        };
    };

    static CppBellmanFord::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppBellmanFord::VertexDescriptor operator()(const CppBellmanFord::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterUnpack(ei);
        };
    };

    typedef read_write_property_map<CppBellmanFord::VertexDescriptor, CppBellmanFord::VertexIterator, CppBellmanFord::VertexDescriptor, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_vertexIterUnpack>::PropertyMap VertexPredecessorMap;
    struct _emptyVPMap {
        inline CppBellmanFord::VertexPredecessorMap operator()() {
            return __read_write_property_map1.emptyMap();
        };
    };

    static CppBellmanFord::_emptyVPMap emptyVPMap;
private:
    static read_write_property_map<CppBellmanFord::VertexDescriptor, CppBellmanFord::VertexIterator, CppBellmanFord::VertexDescriptor, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_vertexIterUnpack> __read_write_property_map1;
public:
    static CppBellmanFord::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex> __incidence_and_vertex_list_and_edge_list_graph;
public:
    typedef base_unit::Unit Unit;
    struct _forIterationEnd {
        inline void operator()(const CppBellmanFord::VertexIterator& itr, const CppBellmanFord::VertexPredecessorMap& state, const CppBellmanFord::Unit& ctx) {
            CppBellmanFord::VertexPredecessorMap mut_state = state;
            if (CppBellmanFord::vertexIterEnd(itr))
            {
                CppBellmanFord::populateVPMapLoopRepeat(itr, mut_state, ctx);
                assert((mut_state) == (state));
            }
            else
                ;
        };
    };

    static CppBellmanFord::_forIterationEnd forIterationEnd;
    struct _populateVPMapLoopRepeat {
        inline void operator()(const CppBellmanFord::VertexIterator& itr, CppBellmanFord::VertexPredecessorMap& state, const CppBellmanFord::Unit& ctx) {
            return __for_iterator_loop.forLoopRepeat(itr, state, ctx);
        };
    };

    static CppBellmanFord::_populateVPMapLoopRepeat populateVPMapLoopRepeat;
    struct _populateVPMapLoopStep {
        inline void operator()(const CppBellmanFord::VertexIterator& itr, CppBellmanFord::VertexPredecessorMap& vpm, const CppBellmanFord::Unit& u) {
            CppBellmanFord::VertexDescriptor v = CppBellmanFord::vertexIterUnpack(itr);
            CppBellmanFord::put(vpm, v, v);
        };
    };

private:
    static for_iterator_loop<CppBellmanFord::Unit, CppBellmanFord::VertexIterator, CppBellmanFord::VertexPredecessorMap, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_populateVPMapLoopStep> __for_iterator_loop;
public:
    static CppBellmanFord::_populateVPMapLoopStep populateVPMapLoopStep;
    struct _unit {
        inline CppBellmanFord::Unit operator()() {
            return __base_unit.unit();
        };
    };

    static CppBellmanFord::_unit unit;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::OutEdgeIterator OutEdgeIterator;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppBellmanFord::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterEnd(ei);
        };
    };

    static CppBellmanFord::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppBellmanFord::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterNext(ei);
        };
    };

    static CppBellmanFord::_outEdgeIterNext outEdgeIterNext;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::Graph Graph;
    struct _numVertices {
        inline CppBellmanFord::VertexCount operator()(const CppBellmanFord::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppBellmanFord::_numVertices numVertices;
    struct _outDegree {
        inline CppBellmanFord::VertexCount operator()(const CppBellmanFord::VertexDescriptor& v, const CppBellmanFord::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppBellmanFord::_outDegree outDegree;
    struct _outEdges {
        inline void operator()(const CppBellmanFord::VertexDescriptor& v, const CppBellmanFord::Graph& g, CppBellmanFord::OutEdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppBellmanFord::_outEdges outEdges;
    struct _toVertexDescriptor {
        inline CppBellmanFord::VertexDescriptor operator()(const CppBellmanFord::Vertex& v, const CppBellmanFord::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppBellmanFord::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppBellmanFord::Graph& g, CppBellmanFord::VertexIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
        };
    };

    static CppBellmanFord::_vertices vertices;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::EdgeIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppBellmanFord::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterEnd(ei);
        };
    };

    static CppBellmanFord::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppBellmanFord::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterNext(ei);
        };
    };

    static CppBellmanFord::_edgeIterNext edgeIterNext;
    struct _edges {
        inline void operator()(const CppBellmanFord::Graph& g, CppBellmanFord::EdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppBellmanFord::_edges edges;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _edgeIterUnpack {
        inline CppBellmanFord::EdgeDescriptor operator()(const CppBellmanFord::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppBellmanFord::_edgeIterUnpack edgeIterUnpack;
    struct _edgeMinimized {
        inline void operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g, CppBellmanFord::Unit& a) {
            ;
        };
    };

    static CppBellmanFord::_edgeMinimized edgeMinimized;
    struct _edgeNotMinimized {
        inline void operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g, CppBellmanFord::Unit& a) {
            ;
        };
    };

    static CppBellmanFord::_edgeNotMinimized edgeNotMinimized;
    struct _edgeNotRelaxed {
        inline void operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g, CppBellmanFord::Unit& a) {
            ;
        };
    };

    static CppBellmanFord::_edgeNotRelaxed edgeNotRelaxed;
    struct _edgeRelaxed {
        inline void operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g, CppBellmanFord::Unit& a) {
            ;
        };
    };

    static CppBellmanFord::_edgeRelaxed edgeRelaxed;
    struct _examineEdge {
        inline void operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g, CppBellmanFord::Unit& a) {
            ;
        };
    };

    static CppBellmanFord::_examineEdge examineEdge;
    struct _outEdgeIterUnpack {
        inline CppBellmanFord::EdgeDescriptor operator()(const CppBellmanFord::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterUnpack(ei);
        };
    };

    static CppBellmanFord::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _src {
        inline CppBellmanFord::VertexDescriptor operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppBellmanFord::_src src;
    struct _tgt {
        inline CppBellmanFord::VertexDescriptor operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppBellmanFord::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppBellmanFord::EdgeDescriptor operator()(const CppBellmanFord::VertexDescriptor& v1, const CppBellmanFord::VertexDescriptor& v2, const CppBellmanFord::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppBellmanFord::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppBellmanFord::Edge operator()(const CppBellmanFord::Vertex& s, const CppBellmanFord::Vertex& t) {
            return __incidence_and_vertex_list_and_edge_list_graph.makeEdge(s, t);
        };
    };

    static CppBellmanFord::_makeEdge makeEdge;
    typedef base_float_ops::Float Cost;
    typedef read_write_property_map<CppBellmanFord::EdgeDescriptor, CppBellmanFord::OutEdgeIterator, CppBellmanFord::Cost, CppBellmanFord::_outEdgeIterEnd, CppBellmanFord::_outEdgeIterNext, CppBellmanFord::_outEdgeIterUnpack>::PropertyMap EdgeCostMap;
    struct _emptyECMap {
        inline CppBellmanFord::EdgeCostMap operator()() {
            return __read_write_property_map.emptyMap();
        };
    };

    static CppBellmanFord::_emptyECMap emptyECMap;
    typedef read_write_property_map<CppBellmanFord::VertexDescriptor, CppBellmanFord::VertexIterator, CppBellmanFord::Cost, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_vertexIterUnpack>::PropertyMap VertexCostMap;
    struct _edgeRelaxationInnerLoopRepeat {
        inline void operator()(const CppBellmanFord::EdgeIterator& itr, CppBellmanFord::Unit& s1, CppBellmanFord::VertexCostMap& s2, CppBellmanFord::VertexPredecessorMap& s3, const CppBellmanFord::EdgeCostMap& ctx1, const CppBellmanFord::Graph& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppBellmanFord::_edgeRelaxationInnerLoopRepeat edgeRelaxationInnerLoopRepeat;
    struct _edgeRelaxationInnerLoopStep {
        inline void operator()(const CppBellmanFord::EdgeIterator& edgeItr, CppBellmanFord::Unit& a, CppBellmanFord::VertexCostMap& vcm, CppBellmanFord::VertexPredecessorMap& vpm, const CppBellmanFord::EdgeCostMap& ecm, const CppBellmanFord::Graph& g) {
            CppBellmanFord::EdgeDescriptor currentEdge = CppBellmanFord::edgeIterUnpack(edgeItr);
            CppBellmanFord::VertexCostMap origVcm = vcm;
            CppBellmanFord::relax(currentEdge, g, ecm, vcm, vpm);
            if ((vcm) == (origVcm))
                CppBellmanFord::edgeRelaxed(currentEdge, g, a);
            else
                CppBellmanFord::edgeNotRelaxed(currentEdge, g, a);
        };
    };

private:
    static for_iterator_loop3_2<CppBellmanFord::EdgeCostMap, CppBellmanFord::Graph, CppBellmanFord::EdgeIterator, CppBellmanFord::Unit, CppBellmanFord::VertexCostMap, CppBellmanFord::VertexPredecessorMap, CppBellmanFord::_edgeIterEnd, CppBellmanFord::_edgeIterNext, CppBellmanFord::_edgeRelaxationInnerLoopStep> __for_iterator_loop3_2;
public:
    static CppBellmanFord::_edgeRelaxationInnerLoopStep edgeRelaxationInnerLoopStep;
    struct _edgeRelaxationOuterLoopRepeat {
        inline void operator()(const CppBellmanFord::VertexIterator& itr, CppBellmanFord::Unit& s1, CppBellmanFord::VertexCostMap& s2, CppBellmanFord::VertexPredecessorMap& s3, const CppBellmanFord::EdgeCostMap& ctx1, const CppBellmanFord::Graph& ctx2) {
            return __for_iterator_loop3_20.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppBellmanFord::_edgeRelaxationOuterLoopRepeat edgeRelaxationOuterLoopRepeat;
    struct _edgeRelaxationOuterLoopStep {
        inline void operator()(const CppBellmanFord::VertexIterator& vertexItr, CppBellmanFord::Unit& a, CppBellmanFord::VertexCostMap& vcm, CppBellmanFord::VertexPredecessorMap& vpm, const CppBellmanFord::EdgeCostMap& ecm, const CppBellmanFord::Graph& g) {
            CppBellmanFord::EdgeIterator edgeItr;
            CppBellmanFord::edges(g, edgeItr);
            CppBellmanFord::edgeRelaxationInnerLoopRepeat(edgeItr, a, vcm, vpm, ecm, g);
        };
    };

private:
    static for_iterator_loop3_2<CppBellmanFord::EdgeCostMap, CppBellmanFord::Graph, CppBellmanFord::VertexIterator, CppBellmanFord::Unit, CppBellmanFord::VertexCostMap, CppBellmanFord::VertexPredecessorMap, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_edgeRelaxationOuterLoopStep> __for_iterator_loop3_20;
public:
    static CppBellmanFord::_edgeRelaxationOuterLoopStep edgeRelaxationOuterLoopStep;
    struct _emptyVCMap {
        inline CppBellmanFord::VertexCostMap operator()() {
            return __read_write_property_map0.emptyMap();
        };
    };

    static CppBellmanFord::_emptyVCMap emptyVCMap;
    struct _relax {
        inline void operator()(const CppBellmanFord::EdgeDescriptor& e, const CppBellmanFord::Graph& g, const CppBellmanFord::EdgeCostMap& ecm, CppBellmanFord::VertexCostMap& vcm, CppBellmanFord::VertexPredecessorMap& vpm) {
            CppBellmanFord::VertexDescriptor u = CppBellmanFord::src(e, g);
            CppBellmanFord::VertexDescriptor v = CppBellmanFord::tgt(e, g);
            CppBellmanFord::Cost uCost = CppBellmanFord::get(vcm, u);
            CppBellmanFord::Cost vCost = CppBellmanFord::get(vcm, v);
            CppBellmanFord::Cost edgeCost = CppBellmanFord::get(ecm, e);
            if (CppBellmanFord::less(CppBellmanFord::plus(uCost, edgeCost), vCost))
            {
                CppBellmanFord::put(vcm, v, CppBellmanFord::plus(uCost, edgeCost));
                CppBellmanFord::put(vpm, v, u);
            }
            else
                ;
        };
    };

    static CppBellmanFord::_relax relax;
private:
    static read_write_property_map<CppBellmanFord::EdgeDescriptor, CppBellmanFord::OutEdgeIterator, CppBellmanFord::Cost, CppBellmanFord::_outEdgeIterEnd, CppBellmanFord::_outEdgeIterNext, CppBellmanFord::_outEdgeIterUnpack> __read_write_property_map;
    static read_write_property_map<CppBellmanFord::VertexDescriptor, CppBellmanFord::VertexIterator, CppBellmanFord::Cost, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_vertexIterUnpack> __read_write_property_map0;
public:
    struct _get {
        inline CppBellmanFord::Cost operator()(const CppBellmanFord::VertexCostMap& pm, const CppBellmanFord::VertexDescriptor& k) {
            return __read_write_property_map0.get(pm, k);
        };
        inline CppBellmanFord::VertexDescriptor operator()(const CppBellmanFord::VertexPredecessorMap& pm, const CppBellmanFord::VertexDescriptor& k) {
            return __read_write_property_map1.get(pm, k);
        };
        inline CppBellmanFord::Cost operator()(const CppBellmanFord::EdgeCostMap& pm, const CppBellmanFord::EdgeDescriptor& k) {
            return __read_write_property_map.get(pm, k);
        };
    };

    static CppBellmanFord::_get get;
    struct _initMap {
        inline CppBellmanFord::VertexCostMap operator()(const CppBellmanFord::VertexIterator& kli, const CppBellmanFord::Cost& v) {
            return __read_write_property_map0.initMap(kli, v);
        };
        inline CppBellmanFord::VertexPredecessorMap operator()(const CppBellmanFord::VertexIterator& kli, const CppBellmanFord::VertexDescriptor& v) {
            return __read_write_property_map1.initMap(kli, v);
        };
        inline CppBellmanFord::EdgeCostMap operator()(const CppBellmanFord::OutEdgeIterator& kli, const CppBellmanFord::Cost& v) {
            return __read_write_property_map.initMap(kli, v);
        };
    };

    static CppBellmanFord::_initMap initMap;
    struct _less {
        inline bool operator()(const CppBellmanFord::Cost& i1, const CppBellmanFord::Cost& i2) {
            return __base_float_ops.less(i1, i2);
        };
    };

    static CppBellmanFord::_less less;
    struct _plus {
        inline CppBellmanFord::Cost operator()(const CppBellmanFord::Cost& i1, const CppBellmanFord::Cost& i2) {
            return __base_float_ops.plus(i1, i2);
        };
    };

    static CppBellmanFord::_plus plus;
    struct _put {
        inline void operator()(CppBellmanFord::VertexCostMap& pm, const CppBellmanFord::VertexDescriptor& k, const CppBellmanFord::Cost& v) {
            return __read_write_property_map0.put(pm, k, v);
        };
        inline void operator()(CppBellmanFord::VertexPredecessorMap& pm, const CppBellmanFord::VertexDescriptor& k, const CppBellmanFord::VertexDescriptor& v) {
            return __read_write_property_map1.put(pm, k, v);
        };
        inline void operator()(CppBellmanFord::EdgeCostMap& pm, const CppBellmanFord::EdgeDescriptor& k, const CppBellmanFord::Cost& v) {
            return __read_write_property_map.put(pm, k, v);
        };
    };

    static CppBellmanFord::_put put;
    typedef base_bool::Bool Bool;
    struct _bellmanFordShortestPaths {
        inline void operator()(const CppBellmanFord::Graph& g, CppBellmanFord::VertexCostMap& vcm, const CppBellmanFord::EdgeCostMap& ecm, CppBellmanFord::Unit& a, CppBellmanFord::VertexPredecessorMap& vpm, CppBellmanFord::Bool& allMinimized) {
            CppBellmanFord::VertexIterator vertexItr;
            CppBellmanFord::vertices(g, vertexItr);
            vpm = CppBellmanFord::emptyVPMap();
            CppBellmanFord::populateVPMapLoopRepeat(vertexItr, vpm, CppBellmanFord::unit());
            CppBellmanFord::edgeRelaxationOuterLoopRepeat(vertexItr, a, vcm, vpm, ecm, g);
            CppBellmanFord::EdgeIterator edgeItr;
            CppBellmanFord::edges(g, edgeItr);
            allMinimized = CppBellmanFord::btrue();
            CppBellmanFord::checkNegativeCycleLoopRepeat(edgeItr, a, allMinimized, vcm, ecm, g);
        };
    };

    static CppBellmanFord::_bellmanFordShortestPaths bellmanFordShortestPaths;
    struct _bfalse {
        inline CppBellmanFord::Bool operator()() {
            return __base_bool.bfalse();
        };
    };

    static CppBellmanFord::_bfalse bfalse;
    struct _btrue {
        inline CppBellmanFord::Bool operator()() {
            return __base_bool.btrue();
        };
    };

    static CppBellmanFord::_btrue btrue;
    struct _checkNegativeCycleLoopRepeat {
        inline void operator()(const CppBellmanFord::EdgeIterator& itr, CppBellmanFord::Unit& s1, CppBellmanFord::Bool& s2, const CppBellmanFord::VertexCostMap& ctx1, const CppBellmanFord::EdgeCostMap& ctx2, const CppBellmanFord::Graph& ctx3) {
            return __for_iterator_loop2_3.forLoopRepeat(itr, s1, s2, ctx1, ctx2, ctx3);
        };
    };

    static CppBellmanFord::_checkNegativeCycleLoopRepeat checkNegativeCycleLoopRepeat;
    struct _checkNegativeCycleLoopStep {
        inline void operator()(const CppBellmanFord::EdgeIterator& edgeItr, CppBellmanFord::Unit& a, CppBellmanFord::Bool& allMinimized, const CppBellmanFord::VertexCostMap& vcm, const CppBellmanFord::EdgeCostMap& ecm, const CppBellmanFord::Graph& g) {
            CppBellmanFord::EdgeDescriptor currentEdge = CppBellmanFord::edgeIterUnpack(edgeItr);
            CppBellmanFord::VertexDescriptor u = CppBellmanFord::src(currentEdge, g);
            CppBellmanFord::VertexDescriptor v = CppBellmanFord::tgt(currentEdge, g);
            CppBellmanFord::Cost uCost = CppBellmanFord::get(vcm, u);
            CppBellmanFord::Cost vCost = CppBellmanFord::get(vcm, v);
            CppBellmanFord::Cost edgeCost = CppBellmanFord::get(ecm, currentEdge);
            if (CppBellmanFord::less(CppBellmanFord::plus(uCost, edgeCost), vCost))
            {
                CppBellmanFord::edgeNotMinimized(currentEdge, g, a);
                allMinimized = CppBellmanFord::bfalse();
            }
            else
                CppBellmanFord::edgeMinimized(currentEdge, g, a);
        };
    };

private:
    static for_iterator_loop2_3<CppBellmanFord::VertexCostMap, CppBellmanFord::EdgeCostMap, CppBellmanFord::Graph, CppBellmanFord::EdgeIterator, CppBellmanFord::Unit, CppBellmanFord::Bool, CppBellmanFord::_edgeIterEnd, CppBellmanFord::_edgeIterNext, CppBellmanFord::_checkNegativeCycleLoopStep> __for_iterator_loop2_3;
public:
    static CppBellmanFord::_checkNegativeCycleLoopStep checkNegativeCycleLoopStep;
    struct _holds {
        inline bool operator()(const CppBellmanFord::Bool& b) {
            return (b) == (CppBellmanFord::btrue());
        };
    };

    static CppBellmanFord::_holds holds;
};
} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
struct CppCustomGraphTypeBFSTestVisitor {
private:
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef iterable_list<CppCustomGraphTypeBFSTestVisitor::Vertex>::ListIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::VertexIterator& itr) {
            return __iterable_list0.iterEnd(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::VertexIterator& itr) {
            return __iterable_list0.iterNext(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_vertexIterNext vertexIterNext;
    typedef iterable_list<CppCustomGraphTypeBFSTestVisitor::Vertex>::List VertexList;
    struct _emptyVertexList {
        inline CppCustomGraphTypeBFSTestVisitor::VertexList operator()() {
            return __iterable_list0.empty();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_emptyVertexList emptyVertexList;
    struct _getVertexIterator {
        inline CppCustomGraphTypeBFSTestVisitor::VertexIterator operator()(const CppCustomGraphTypeBFSTestVisitor::VertexList& itb) {
            return __iterable_list0.getIterator(itb);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_getVertexIterator getVertexIterator;
    typedef vector<CppCustomGraphTypeBFSTestVisitor::Vertex>::Vector VertexVector;
    struct _emptyVertexVector {
        inline CppCustomGraphTypeBFSTestVisitor::VertexVector operator()() {
            return __vector.empty();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_emptyVertexVector emptyVertexVector;
private:
    static edge_without_descriptor<CppCustomGraphTypeBFSTestVisitor::Vertex> __edge_without_descriptor;
    static fifo_queue<CppCustomGraphTypeBFSTestVisitor::Vertex> __fifo_queue;
    static iterable_list<CppCustomGraphTypeBFSTestVisitor::Vertex> __iterable_list0;
    static vector<CppCustomGraphTypeBFSTestVisitor::Vertex> __vector;
public:
    struct _pushBack {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& a, CppCustomGraphTypeBFSTestVisitor::VertexVector& v) {
            return __vector.pushBack(a, v);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_pushBack pushBack;
    struct _vertexIterUnpack {
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::VertexIterator& itr) {
            return __iterable_list0.iterUnpack(itr);
        };
    };

private:
    static two_bit_color_map<CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
    typedef base_types::Int Int;
    typedef fifo_queue<CppCustomGraphTypeBFSTestVisitor::Vertex>::FIFOQueue FIFOQueue;
    struct _empty {
        inline CppCustomGraphTypeBFSTestVisitor::FIFOQueue operator()() {
            return __fifo_queue.empty();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_empty empty;
    struct _front {
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.front(q);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_front front;
    struct _isEmptyQueue {
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.isEmpty(q);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_isEmptyQueue isEmptyQueue;
    struct _pop {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.pop(q);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_pop pop;
    struct _push {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& a, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q) {
            return __fifo_queue.push(a, q);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_push push;
    struct _pushPopBehavior {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& a, const CppCustomGraphTypeBFSTestVisitor::FIFOQueue& inq) {
            CppCustomGraphTypeBFSTestVisitor::FIFOQueue mut_inq = inq;
            CppCustomGraphTypeBFSTestVisitor::push(a, mut_inq);
            assert((CppCustomGraphTypeBFSTestVisitor::front(mut_inq)) == (a));
            CppCustomGraphTypeBFSTestVisitor::pop(mut_inq);
            assert((inq) == (mut_inq));
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_pushPopBehavior pushPopBehavior;
    typedef edge_without_descriptor<CppCustomGraphTypeBFSTestVisitor::Vertex>::Edge Edge;
    typedef iterable_list<CppCustomGraphTypeBFSTestVisitor::Edge>::ListIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::EdgeIterator& itr) {
            return __iterable_list.iterEnd(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::EdgeIterator& itr) {
            return __iterable_list.iterNext(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_edgeIterNext edgeIterNext;
    typedef iterable_list<CppCustomGraphTypeBFSTestVisitor::Edge>::List EdgeList;
    struct _emptyEdgeList {
        inline CppCustomGraphTypeBFSTestVisitor::EdgeList operator()() {
            return __iterable_list.empty();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_emptyEdgeList emptyEdgeList;
    struct _getEdgeIterator {
        inline CppCustomGraphTypeBFSTestVisitor::EdgeIterator operator()(const CppCustomGraphTypeBFSTestVisitor::EdgeList& itb) {
            return __iterable_list.getIterator(itb);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_getEdgeIterator getEdgeIterator;
    struct _isEmpty {
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::EdgeList& l) {
            return __iterable_list.isEmpty(l);
        };
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::VertexList& l) {
            return __iterable_list0.isEmpty(l);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_isEmpty isEmpty;
    struct _tail {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::EdgeList& l) {
            return __iterable_list.tail(l);
        };
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::VertexList& l) {
            return __iterable_list0.tail(l);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_tail tail;
    typedef iterable_list<CppCustomGraphTypeBFSTestVisitor::Edge>::ListIterator OutEdgeIterator;
    typedef pair<CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator>::Pair OutEdgeIteratorRange;
private:
    static pair<CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator> __pair;
public:
    struct _getOutEdgeIterator {
        inline CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator operator()(const CppCustomGraphTypeBFSTestVisitor::EdgeList& itb) {
            return __iterable_list.getIterator(itb);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_getOutEdgeIterator getOutEdgeIterator;
    struct _iterRangeBegin {
        inline CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeOutEdgeIteratorRange {
        inline CppCustomGraphTypeBFSTestVisitor::OutEdgeIteratorRange operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& a, const CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_makeOutEdgeIteratorRange makeOutEdgeIteratorRange;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& itr) {
            return __iterable_list.iterEnd(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& itr) {
            return __iterable_list.iterNext(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext outEdgeIterNext;
private:
    static iterable_list<CppCustomGraphTypeBFSTestVisitor::Edge> __iterable_list;
public:
    struct _cons {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Edge& a, CppCustomGraphTypeBFSTestVisitor::EdgeList& l) {
            return __iterable_list.cons(a, l);
        };
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& a, CppCustomGraphTypeBFSTestVisitor::VertexList& l) {
            return __iterable_list0.cons(a, l);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_cons cons;
    struct _edgeIterUnpack {
        inline CppCustomGraphTypeBFSTestVisitor::Edge operator()(const CppCustomGraphTypeBFSTestVisitor::EdgeIterator& itr) {
            return __iterable_list.iterUnpack(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _head {
        inline CppCustomGraphTypeBFSTestVisitor::Edge operator()(const CppCustomGraphTypeBFSTestVisitor::EdgeList& l) {
            return __iterable_list.head(l);
        };
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::VertexList& l) {
            return __iterable_list0.head(l);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_head head;
    struct _makeEdge {
        inline CppCustomGraphTypeBFSTestVisitor::Edge operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& s, const CppCustomGraphTypeBFSTestVisitor::Vertex& t) {
            return __edge_without_descriptor.makeEdge(s, t);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_makeEdge makeEdge;
    struct _outEdgeIterUnpack {
        inline CppCustomGraphTypeBFSTestVisitor::Edge operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& itr) {
            return __iterable_list.iterUnpack(itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _srcPlainEdge {
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::Edge& e) {
            return __edge_without_descriptor.srcPlainEdge(e);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_srcPlainEdge srcPlainEdge;
    struct _tgtPlainEdge {
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::Edge& e) {
            return __edge_without_descriptor.tgtPlainEdge(e);
        };
    };

    typedef custom_incidence_and_vertex_list_and_edge_list_graph<CppCustomGraphTypeBFSTestVisitor::Edge, CppCustomGraphTypeBFSTestVisitor::EdgeIterator, CppCustomGraphTypeBFSTestVisitor::EdgeList, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::VertexList, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_edgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_edgeIterNext, CppCustomGraphTypeBFSTestVisitor::_edgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_emptyEdgeList, CppCustomGraphTypeBFSTestVisitor::_emptyVertexList, CppCustomGraphTypeBFSTestVisitor::_getEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getOutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getVertexIterator, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_makeEdge, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_srcPlainEdge, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tgtPlainEdge, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack>::Graph Graph;
    struct _breadthFirstSearch {
        inline CppCustomGraphTypeBFSTestVisitor::VertexVector operator()(const CppCustomGraphTypeBFSTestVisitor::Graph& g, const CppCustomGraphTypeBFSTestVisitor::Vertex& start, const CppCustomGraphTypeBFSTestVisitor::VertexVector& init) {
            CppCustomGraphTypeBFSTestVisitor::FIFOQueue q = CppCustomGraphTypeBFSTestVisitor::empty();
            CppCustomGraphTypeBFSTestVisitor::VertexIterator vertexItr;
            CppCustomGraphTypeBFSTestVisitor::vertices(g, vertexItr);
            CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap c = CppCustomGraphTypeBFSTestVisitor::initMap(vertexItr, CppCustomGraphTypeBFSTestVisitor::white());
            CppCustomGraphTypeBFSTestVisitor::VertexVector a = init;
            CppCustomGraphTypeBFSTestVisitor::breadthFirstVisit(g, start, a, q, c);
            return a;
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_breadthFirstSearch breadthFirstSearch;
    struct _defaultAction {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& edgeOrVertex, const CppCustomGraphTypeBFSTestVisitor::Graph& g, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, CppCustomGraphTypeBFSTestVisitor::VertexVector& a) {
            ;
        };
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Edge& edgeOrVertex, const CppCustomGraphTypeBFSTestVisitor::Graph& g, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, CppCustomGraphTypeBFSTestVisitor::VertexVector& a) {
            ;
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_defaultAction defaultAction;
    struct _edges {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Graph& g, CppCustomGraphTypeBFSTestVisitor::EdgeIterator& itr) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_edges edges;
    struct _examineVertex {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& v, const CppCustomGraphTypeBFSTestVisitor::Graph& g, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, CppCustomGraphTypeBFSTestVisitor::VertexVector& a) {
            CppCustomGraphTypeBFSTestVisitor::pushBack(v, a);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_examineVertex examineVertex;
    struct _outEdges {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& v, const CppCustomGraphTypeBFSTestVisitor::Graph& g, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& itr) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_outEdges outEdges;
    struct _src {
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::Edge& e, const CppCustomGraphTypeBFSTestVisitor::Graph& g) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_src src;
    struct _tgt {
        inline CppCustomGraphTypeBFSTestVisitor::Vertex operator()(const CppCustomGraphTypeBFSTestVisitor::Edge& e, const CppCustomGraphTypeBFSTestVisitor::Graph& g) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_tgt tgt;
    struct _vertices {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Graph& g, CppCustomGraphTypeBFSTestVisitor::VertexIterator& itr) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_vertices vertices;
    typedef custom_incidence_and_vertex_list_and_edge_list_graph<CppCustomGraphTypeBFSTestVisitor::Edge, CppCustomGraphTypeBFSTestVisitor::EdgeIterator, CppCustomGraphTypeBFSTestVisitor::EdgeList, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::VertexList, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_edgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_edgeIterNext, CppCustomGraphTypeBFSTestVisitor::_edgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_emptyEdgeList, CppCustomGraphTypeBFSTestVisitor::_emptyVertexList, CppCustomGraphTypeBFSTestVisitor::_getEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getOutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getVertexIterator, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_makeEdge, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_srcPlainEdge, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tgtPlainEdge, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack>::VertexCount VertexCount;
    struct _numVertices {
        inline CppCustomGraphTypeBFSTestVisitor::VertexCount operator()(const CppCustomGraphTypeBFSTestVisitor::Graph& g) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppCustomGraphTypeBFSTestVisitor::VertexCount operator()(const CppCustomGraphTypeBFSTestVisitor::Vertex& v, const CppCustomGraphTypeBFSTestVisitor::Graph& g) {
            return __custom_incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_outDegree outDegree;
private:
    static custom_incidence_and_vertex_list_and_edge_list_graph<CppCustomGraphTypeBFSTestVisitor::Edge, CppCustomGraphTypeBFSTestVisitor::EdgeIterator, CppCustomGraphTypeBFSTestVisitor::EdgeList, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::VertexList, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_edgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_edgeIterNext, CppCustomGraphTypeBFSTestVisitor::_edgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_emptyEdgeList, CppCustomGraphTypeBFSTestVisitor::_emptyVertexList, CppCustomGraphTypeBFSTestVisitor::_getEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getOutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getVertexIterator, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_makeEdge, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_srcPlainEdge, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tgtPlainEdge, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack> __custom_incidence_and_vertex_list_and_edge_list_graph;
public:
    static CppCustomGraphTypeBFSTestVisitor::_tgtPlainEdge tgtPlainEdge;
    typedef two_bit_color_map<CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& itr, CppCustomGraphTypeBFSTestVisitor::VertexVector& s1, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& s2, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& s3, const CppCustomGraphTypeBFSTestVisitor::Graph& ctx1, const CppCustomGraphTypeBFSTestVisitor::Vertex& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator& edgeItr, CppCustomGraphTypeBFSTestVisitor::VertexVector& x, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& c, const CppCustomGraphTypeBFSTestVisitor::Graph& g, const CppCustomGraphTypeBFSTestVisitor::Vertex& u) {
            CppCustomGraphTypeBFSTestVisitor::Edge e = CppCustomGraphTypeBFSTestVisitor::outEdgeIterUnpack(edgeItr);
            CppCustomGraphTypeBFSTestVisitor::Vertex v = CppCustomGraphTypeBFSTestVisitor::tgt(e, g);
            CppCustomGraphTypeBFSTestVisitor::defaultAction(e, g, q, x);
            CppCustomGraphTypeBFSTestVisitor::Color vc = CppCustomGraphTypeBFSTestVisitor::get(c, v);
            if ((vc) == (CppCustomGraphTypeBFSTestVisitor::white()))
            {
                CppCustomGraphTypeBFSTestVisitor::defaultAction(e, g, q, x);
                CppCustomGraphTypeBFSTestVisitor::put(c, v, CppCustomGraphTypeBFSTestVisitor::gray());
                CppCustomGraphTypeBFSTestVisitor::defaultAction(v, g, q, x);
                CppCustomGraphTypeBFSTestVisitor::push(v, q);
            }
            else
                if ((vc) == (CppCustomGraphTypeBFSTestVisitor::gray()))
                {
                    CppCustomGraphTypeBFSTestVisitor::defaultAction(e, g, q, x);
                }
                else
                {
                    CppCustomGraphTypeBFSTestVisitor::defaultAction(e, g, q, x);
                }
        };
    };

private:
    static for_iterator_loop3_2<CppCustomGraphTypeBFSTestVisitor::Graph, CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::VertexVector, CppCustomGraphTypeBFSTestVisitor::FIFOQueue, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext, CppCustomGraphTypeBFSTestVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
public:
    static CppCustomGraphTypeBFSTestVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
    struct _bfsOuterLoopCond {
        inline bool operator()(const CppCustomGraphTypeBFSTestVisitor::VertexVector& a, const CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, const CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& c, const CppCustomGraphTypeBFSTestVisitor::Graph& g) {
            return !CppCustomGraphTypeBFSTestVisitor::isEmptyQueue(q);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::VertexVector& s1, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& s2, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& s3, const CppCustomGraphTypeBFSTestVisitor::Graph& ctx) {
            return __while_loop3.repeat(s1, s2, s3, ctx);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::VertexVector& x, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& c, const CppCustomGraphTypeBFSTestVisitor::Graph& g) {
            CppCustomGraphTypeBFSTestVisitor::Vertex u = CppCustomGraphTypeBFSTestVisitor::front(q);
            CppCustomGraphTypeBFSTestVisitor::pop(q);
            CppCustomGraphTypeBFSTestVisitor::examineVertex(u, g, q, x);
            CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator edgeItr;
            CppCustomGraphTypeBFSTestVisitor::outEdges(u, g, edgeItr);
            CppCustomGraphTypeBFSTestVisitor::bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);
            CppCustomGraphTypeBFSTestVisitor::put(c, u, CppCustomGraphTypeBFSTestVisitor::black());
            CppCustomGraphTypeBFSTestVisitor::defaultAction(u, g, q, x);
        };
    };

private:
    static while_loop3<CppCustomGraphTypeBFSTestVisitor::Graph, CppCustomGraphTypeBFSTestVisitor::VertexVector, CppCustomGraphTypeBFSTestVisitor::FIFOQueue, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap, CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopCond, CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopStep> __while_loop3;
public:
    static CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _breadthFirstVisit {
        inline void operator()(const CppCustomGraphTypeBFSTestVisitor::Graph& g, const CppCustomGraphTypeBFSTestVisitor::Vertex& s, CppCustomGraphTypeBFSTestVisitor::VertexVector& a, CppCustomGraphTypeBFSTestVisitor::FIFOQueue& q, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& c) {
            CppCustomGraphTypeBFSTestVisitor::defaultAction(s, g, q, a);
            CppCustomGraphTypeBFSTestVisitor::push(s, q);
            CppCustomGraphTypeBFSTestVisitor::put(c, s, CppCustomGraphTypeBFSTestVisitor::gray());
            CppCustomGraphTypeBFSTestVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_breadthFirstVisit breadthFirstVisit;
    typedef two_bit_color_map<CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack>::Color Color;
    struct _black {
        inline CppCustomGraphTypeBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.black();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_black black;
    struct _get {
        inline CppCustomGraphTypeBFSTestVisitor::Color operator()(const CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& pm, const CppCustomGraphTypeBFSTestVisitor::Vertex& k) {
            return __two_bit_color_map.get(pm, k);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_get get;
    struct _gray {
        inline CppCustomGraphTypeBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.gray();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_gray gray;
    struct _initMap {
        inline CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap operator()(const CppCustomGraphTypeBFSTestVisitor::VertexIterator& kli, const CppCustomGraphTypeBFSTestVisitor::Color& v) {
            return __two_bit_color_map.initMap(kli, v);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_initMap initMap;
    struct _put {
        inline void operator()(CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap& pm, const CppCustomGraphTypeBFSTestVisitor::Vertex& k, const CppCustomGraphTypeBFSTestVisitor::Color& v) {
            return __two_bit_color_map.put(pm, k, v);
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_put put;
    struct _white {
        inline CppCustomGraphTypeBFSTestVisitor::Color operator()() {
            return __two_bit_color_map.white();
        };
    };

    static CppCustomGraphTypeBFSTestVisitor::_white white;
};
} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::VertexDescriptor VertexDescriptor;
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppDFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterEnd(ei);
        };
    };

    static CppDFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppDFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterNext(ei);
        };
    };

    static CppDFSTestVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterUnpack(ei);
        };
    };

private:
    static two_bit_color_map<CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::VertexIterator, CppDFSTestVisitor::_vertexIterEnd, CppDFSTestVisitor::_vertexIterNext, CppDFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppDFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex> __incidence_and_vertex_list_and_edge_list_graph;
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::OutEdgeIterator OutEdgeIterator;
    typedef pair<CppDFSTestVisitor::OutEdgeIterator, CppDFSTestVisitor::OutEdgeIterator>::Pair OutEdgeIteratorRange;
private:
    static pair<CppDFSTestVisitor::OutEdgeIterator, CppDFSTestVisitor::OutEdgeIterator> __pair;
public:
    struct _iterRangeBegin {
        inline CppDFSTestVisitor::OutEdgeIterator operator()(const CppDFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppDFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppDFSTestVisitor::OutEdgeIterator operator()(const CppDFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppDFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeOutEdgeIteratorRange {
        inline CppDFSTestVisitor::OutEdgeIteratorRange operator()(const CppDFSTestVisitor::OutEdgeIterator& a, const CppDFSTestVisitor::OutEdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppDFSTestVisitor::_makeOutEdgeIteratorRange makeOutEdgeIteratorRange;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppDFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterEnd(ei);
        };
    };

    static CppDFSTestVisitor::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppDFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterNext(ei);
        };
    };

    static CppDFSTestVisitor::_outEdgeIterNext outEdgeIterNext;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::Graph Graph;
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
            return __incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppDFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppDFSTestVisitor::VertexCount operator()(const CppDFSTestVisitor::VertexDescriptor& v, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppDFSTestVisitor::_outDegree outDegree;
    struct _outEdges {
        inline void operator()(const CppDFSTestVisitor::VertexDescriptor& v, const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::OutEdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppDFSTestVisitor::_outEdges outEdges;
    struct _toVertexDescriptor {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::Vertex& v, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppDFSTestVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
        };
    };

    static CppDFSTestVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::EdgeIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppDFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterEnd(ei);
        };
    };

    static CppDFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppDFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterNext(ei);
        };
    };

    static CppDFSTestVisitor::_edgeIterNext edgeIterNext;
    struct _edges {
        inline void operator()(const CppDFSTestVisitor::Graph& g, CppDFSTestVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppDFSTestVisitor::_edges edges;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
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
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppDFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _outEdgeIterUnpack {
        inline CppDFSTestVisitor::EdgeDescriptor operator()(const CppDFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterUnpack(ei);
        };
    };

    static CppDFSTestVisitor::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _src {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::EdgeDescriptor& e, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppDFSTestVisitor::_src src;
    struct _tgt {
        inline CppDFSTestVisitor::VertexDescriptor operator()(const CppDFSTestVisitor::EdgeDescriptor& e, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppDFSTestVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppDFSTestVisitor::EdgeDescriptor operator()(const CppDFSTestVisitor::VertexDescriptor& v1, const CppDFSTestVisitor::VertexDescriptor& v2, const CppDFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppDFSTestVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppDFSTestVisitor::Edge operator()(const CppDFSTestVisitor::Vertex& s, const CppDFSTestVisitor::Vertex& t) {
            return __incidence_and_vertex_list_and_edge_list_graph.makeEdge(s, t);
        };
    };

    static CppDFSTestVisitor::_makeEdge makeEdge;
    typedef two_bit_color_map<CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::VertexIterator, CppDFSTestVisitor::_vertexIterEnd, CppDFSTestVisitor::_vertexIterNext, CppDFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppDFSTestVisitor::OutEdgeIterator& itr, CppDFSTestVisitor::VertexVector& s1, CppDFSTestVisitor::Stack& s2, CppDFSTestVisitor::ColorPropertyMap& s3, const CppDFSTestVisitor::Graph& ctx1, const CppDFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppDFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppDFSTestVisitor::OutEdgeIterator& edgeItr, CppDFSTestVisitor::VertexVector& x, CppDFSTestVisitor::Stack& q, CppDFSTestVisitor::ColorPropertyMap& c, const CppDFSTestVisitor::Graph& g, const CppDFSTestVisitor::VertexDescriptor& u) {
            CppDFSTestVisitor::EdgeDescriptor e = CppDFSTestVisitor::outEdgeIterUnpack(edgeItr);
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
    static for_iterator_loop3_2<CppDFSTestVisitor::Graph, CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::OutEdgeIterator, CppDFSTestVisitor::VertexVector, CppDFSTestVisitor::Stack, CppDFSTestVisitor::ColorPropertyMap, CppDFSTestVisitor::_outEdgeIterEnd, CppDFSTestVisitor::_outEdgeIterNext, CppDFSTestVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
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
            CppDFSTestVisitor::OutEdgeIterator edgeItr;
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
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
struct CppDijkstraVisitor {
private:
    static base_types __base_types;
    static base_float_ops __base_float_ops;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::VertexDescriptor VertexDescriptor;
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppDijkstraVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterEnd(ei);
        };
    };

    static CppDijkstraVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppDijkstraVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterNext(ei);
        };
    };

    static CppDijkstraVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterUnpack(ei);
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
    static incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex> __incidence_and_vertex_list_and_edge_list_graph;
public:
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::OutEdgeIterator OutEdgeIterator;
    typedef pair<CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::OutEdgeIterator>::Pair OutEdgeIteratorRange;
private:
    static pair<CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::OutEdgeIterator> __pair;
public:
    struct _iterRangeBegin {
        inline CppDijkstraVisitor::OutEdgeIterator operator()(const CppDijkstraVisitor::OutEdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppDijkstraVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppDijkstraVisitor::OutEdgeIterator operator()(const CppDijkstraVisitor::OutEdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppDijkstraVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeOutEdgeIteratorRange {
        inline CppDijkstraVisitor::OutEdgeIteratorRange operator()(const CppDijkstraVisitor::OutEdgeIterator& a, const CppDijkstraVisitor::OutEdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppDijkstraVisitor::_makeOutEdgeIteratorRange makeOutEdgeIteratorRange;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppDijkstraVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterEnd(ei);
        };
    };

    static CppDijkstraVisitor::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppDijkstraVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterNext(ei);
        };
    };

    static CppDijkstraVisitor::_outEdgeIterNext outEdgeIterNext;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::Graph Graph;
    struct _numVertices {
        inline CppDijkstraVisitor::VertexCount operator()(const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppDijkstraVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppDijkstraVisitor::VertexCount operator()(const CppDijkstraVisitor::VertexDescriptor& v, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppDijkstraVisitor::_outDegree outDegree;
    struct _outEdges {
        inline void operator()(const CppDijkstraVisitor::VertexDescriptor& v, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::OutEdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppDijkstraVisitor::_outEdges outEdges;
    struct _toVertexDescriptor {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::Vertex& v, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppDijkstraVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
        };
    };

    static CppDijkstraVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::EdgeIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppDijkstraVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterEnd(ei);
        };
    };

    static CppDijkstraVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppDijkstraVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterNext(ei);
        };
    };

    static CppDijkstraVisitor::_edgeIterNext edgeIterNext;
    struct _edges {
        inline void operator()(const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppDijkstraVisitor::_edges edges;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _edgeIterUnpack {
        inline CppDijkstraVisitor::EdgeDescriptor operator()(const CppDijkstraVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppDijkstraVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _outEdgeIterUnpack {
        inline CppDijkstraVisitor::EdgeDescriptor operator()(const CppDijkstraVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterUnpack(ei);
        };
    };

    static CppDijkstraVisitor::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _src {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppDijkstraVisitor::_src src;
    struct _tgt {
        inline CppDijkstraVisitor::VertexDescriptor operator()(const CppDijkstraVisitor::EdgeDescriptor& e, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppDijkstraVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppDijkstraVisitor::EdgeDescriptor operator()(const CppDijkstraVisitor::VertexDescriptor& v1, const CppDijkstraVisitor::VertexDescriptor& v2, const CppDijkstraVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppDijkstraVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppDijkstraVisitor::Edge operator()(const CppDijkstraVisitor::Vertex& s, const CppDijkstraVisitor::Vertex& t) {
            return __incidence_and_vertex_list_and_edge_list_graph.makeEdge(s, t);
        };
    };

    static CppDijkstraVisitor::_makeEdge makeEdge;
    typedef base_float_ops::Float Cost;
    typedef read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_outEdgeIterEnd, CppDijkstraVisitor::_outEdgeIterNext, CppDijkstraVisitor::_outEdgeIterUnpack>::PropertyMap EdgeCostMap;
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
    static read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_outEdgeIterEnd, CppDijkstraVisitor::_outEdgeIterNext, CppDijkstraVisitor::_outEdgeIterUnpack> __read_write_property_map;
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
        inline void operator()(const CppDijkstraVisitor::OutEdgeIterator& itr, CppDijkstraVisitor::StateWithMaps& s1, CppDijkstraVisitor::PriorityQueue& s2, CppDijkstraVisitor::ColorPropertyMap& s3, const CppDijkstraVisitor::Graph& ctx1, const CppDijkstraVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppDijkstraVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppDijkstraVisitor::OutEdgeIterator& edgeItr, CppDijkstraVisitor::StateWithMaps& x, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::ColorPropertyMap& c, const CppDijkstraVisitor::Graph& g, const CppDijkstraVisitor::VertexDescriptor& u) {
            CppDijkstraVisitor::EdgeDescriptor e = CppDijkstraVisitor::outEdgeIterUnpack(edgeItr);
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
    static for_iterator_loop3_2<CppDijkstraVisitor::Graph, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap, CppDijkstraVisitor::_outEdgeIterEnd, CppDijkstraVisitor::_outEdgeIterNext, CppDijkstraVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
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
            CppDijkstraVisitor::OutEdgeIterator edgeItr;
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
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
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
        inline void operator()(const CppDijkstraVisitor::EdgeDescriptor& edgeOrVertex, const CppDijkstraVisitor::Graph& g, CppDijkstraVisitor::PriorityQueue& q, CppDijkstraVisitor::StateWithMaps& a) {
            ;
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
        inline CppDijkstraVisitor::EdgeCostMap operator()(const CppDijkstraVisitor::OutEdgeIterator& kli, const CppDijkstraVisitor::Cost& v) {
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
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
struct CppParallelBFSTestVisitor {
private:
    static base_types __base_types;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::VertexDescriptor VertexDescriptor;
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppParallelBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterEnd(ei);
        };
    };

    static CppParallelBFSTestVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppParallelBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterNext(ei);
        };
    };

    static CppParallelBFSTestVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterUnpack(ei);
        };
    };

private:
    static two_bit_color_map<CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::VertexIterator, CppParallelBFSTestVisitor::_vertexIterEnd, CppParallelBFSTestVisitor::_vertexIterNext, CppParallelBFSTestVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppParallelBFSTestVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex> __incidence_and_vertex_list_and_edge_list_graph;
public:
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::OutEdgeIterator OutEdgeIterator;
    typedef pair<CppParallelBFSTestVisitor::OutEdgeIterator, CppParallelBFSTestVisitor::OutEdgeIterator>::Pair OutEdgeIteratorRange;
private:
    static pair<CppParallelBFSTestVisitor::OutEdgeIterator, CppParallelBFSTestVisitor::OutEdgeIterator> __pair;
public:
    struct _iterRangeBegin {
        inline CppParallelBFSTestVisitor::OutEdgeIterator operator()(const CppParallelBFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppParallelBFSTestVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppParallelBFSTestVisitor::OutEdgeIterator operator()(const CppParallelBFSTestVisitor::OutEdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppParallelBFSTestVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeOutEdgeIteratorRange {
        inline CppParallelBFSTestVisitor::OutEdgeIteratorRange operator()(const CppParallelBFSTestVisitor::OutEdgeIterator& a, const CppParallelBFSTestVisitor::OutEdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppParallelBFSTestVisitor::_makeOutEdgeIteratorRange makeOutEdgeIteratorRange;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppParallelBFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterEnd(ei);
        };
    };

    static CppParallelBFSTestVisitor::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppParallelBFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterNext(ei);
        };
    };

    static CppParallelBFSTestVisitor::_outEdgeIterNext outEdgeIterNext;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::Graph Graph;
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
            return __incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppParallelBFSTestVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppParallelBFSTestVisitor::VertexCount operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppParallelBFSTestVisitor::_outDegree outDegree;
    struct _outEdges {
        inline void operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v, const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::OutEdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppParallelBFSTestVisitor::_outEdges outEdges;
    struct _toVertexDescriptor {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::Vertex& v, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppParallelBFSTestVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
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
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::EdgeIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppParallelBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterEnd(ei);
        };
    };

    static CppParallelBFSTestVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppParallelBFSTestVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterNext(ei);
        };
    };

    static CppParallelBFSTestVisitor::_edgeIterNext edgeIterNext;
    struct _edges {
        inline void operator()(const CppParallelBFSTestVisitor::Graph& g, CppParallelBFSTestVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppParallelBFSTestVisitor::_edges edges;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
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
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppParallelBFSTestVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _outEdgeIterUnpack {
        inline CppParallelBFSTestVisitor::EdgeDescriptor operator()(const CppParallelBFSTestVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterUnpack(ei);
        };
    };

    static CppParallelBFSTestVisitor::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _src {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::EdgeDescriptor& e, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppParallelBFSTestVisitor::_src src;
    struct _tgt {
        inline CppParallelBFSTestVisitor::VertexDescriptor operator()(const CppParallelBFSTestVisitor::EdgeDescriptor& e, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppParallelBFSTestVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppParallelBFSTestVisitor::EdgeDescriptor operator()(const CppParallelBFSTestVisitor::VertexDescriptor& v1, const CppParallelBFSTestVisitor::VertexDescriptor& v2, const CppParallelBFSTestVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppParallelBFSTestVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppParallelBFSTestVisitor::Edge operator()(const CppParallelBFSTestVisitor::Vertex& s, const CppParallelBFSTestVisitor::Vertex& t) {
            return __incidence_and_vertex_list_and_edge_list_graph.makeEdge(s, t);
        };
    };

    static CppParallelBFSTestVisitor::_makeEdge makeEdge;
    typedef two_bit_color_map<CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::VertexIterator, CppParallelBFSTestVisitor::_vertexIterEnd, CppParallelBFSTestVisitor::_vertexIterNext, CppParallelBFSTestVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppParallelBFSTestVisitor::OutEdgeIterator& itr, CppParallelBFSTestVisitor::VertexVector& s1, CppParallelBFSTestVisitor::FIFOQueue& s2, CppParallelBFSTestVisitor::ColorPropertyMap& s3, const CppParallelBFSTestVisitor::Graph& ctx1, const CppParallelBFSTestVisitor::VertexDescriptor& ctx2) {
            return __for_parallel_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppParallelBFSTestVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppParallelBFSTestVisitor::OutEdgeIterator& edgeItr, CppParallelBFSTestVisitor::VertexVector& x, CppParallelBFSTestVisitor::FIFOQueue& q, CppParallelBFSTestVisitor::ColorPropertyMap& c, const CppParallelBFSTestVisitor::Graph& g, const CppParallelBFSTestVisitor::VertexDescriptor& u) {
            CppParallelBFSTestVisitor::EdgeDescriptor e = CppParallelBFSTestVisitor::outEdgeIterUnpack(edgeItr);
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
    static for_parallel_iterator_loop3_2<CppParallelBFSTestVisitor::Graph, CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::OutEdgeIterator, CppParallelBFSTestVisitor::VertexVector, CppParallelBFSTestVisitor::FIFOQueue, CppParallelBFSTestVisitor::ColorPropertyMap, CppParallelBFSTestVisitor::_outEdgeIterEnd, CppParallelBFSTestVisitor::_outEdgeIterNext, CppParallelBFSTestVisitor::_bfsInnerLoopStep> __for_parallel_iterator_loop3_2;
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
            CppParallelBFSTestVisitor::OutEdgeIterator edgeItr;
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
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
struct CppPrimVisitor {
private:
    static base_types __base_types;
    static base_float_ops __base_float_ops;
public:
    typedef base_types::Vertex Vertex;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::VertexCount VertexCount;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::VertexDescriptor VertexDescriptor;
    typedef vector<CppPrimVisitor::VertexDescriptor>::Vector VertexVector;
    struct _emptyVertexVector {
        inline CppPrimVisitor::VertexVector operator()() {
            return __vector.empty();
        };
    };

    static CppPrimVisitor::_emptyVertexVector emptyVertexVector;
private:
    static vector<CppPrimVisitor::VertexDescriptor> __vector;
public:
    struct _pushBack {
        inline void operator()(const CppPrimVisitor::VertexDescriptor& a, CppPrimVisitor::VertexVector& v) {
            return __vector.pushBack(a, v);
        };
    };

    static CppPrimVisitor::_pushBack pushBack;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::VertexIterator VertexIterator;
    struct _vertexIterEnd {
        inline bool operator()(const CppPrimVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterEnd(ei);
        };
    };

    static CppPrimVisitor::_vertexIterEnd vertexIterEnd;
    struct _vertexIterNext {
        inline void operator()(CppPrimVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterNext(ei);
        };
    };

    static CppPrimVisitor::_vertexIterNext vertexIterNext;
    struct _vertexIterUnpack {
        inline CppPrimVisitor::VertexDescriptor operator()(const CppPrimVisitor::VertexIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertexIterUnpack(ei);
        };
    };

    typedef read_write_property_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::VertexDescriptor, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack>::PropertyMap VertexPredecessorMap;
    struct _emptyVPMap {
        inline CppPrimVisitor::VertexPredecessorMap operator()() {
            return __read_write_property_map1.emptyMap();
        };
    };

    static CppPrimVisitor::_emptyVPMap emptyVPMap;
    struct _forIterationEnd {
        inline void operator()(const CppPrimVisitor::VertexIterator& itr, const CppPrimVisitor::VertexPredecessorMap& state, const CppPrimVisitor::VertexDescriptor& ctx) {
            CppPrimVisitor::VertexPredecessorMap mut_state = state;
            if (CppPrimVisitor::vertexIterEnd(itr))
            {
                CppPrimVisitor::populateVPMapLoopRepeat(itr, mut_state, ctx);
                assert((mut_state) == (state));
            }
            else
                ;
        };
    };

    static CppPrimVisitor::_forIterationEnd forIterationEnd;
    struct _populateVPMapLoopRepeat {
        inline void operator()(const CppPrimVisitor::VertexIterator& itr, CppPrimVisitor::VertexPredecessorMap& state, const CppPrimVisitor::VertexDescriptor& ctx) {
            return __for_iterator_loop.forLoopRepeat(itr, state, ctx);
        };
    };

    static CppPrimVisitor::_populateVPMapLoopRepeat populateVPMapLoopRepeat;
    struct _populateVPMapLoopStep {
        inline void operator()(const CppPrimVisitor::VertexIterator& itr, CppPrimVisitor::VertexPredecessorMap& vpm, const CppPrimVisitor::VertexDescriptor& vd) {
            CppPrimVisitor::VertexDescriptor v = CppPrimVisitor::vertexIterUnpack(itr);
            CppPrimVisitor::put(vpm, v, v);
        };
    };

private:
    static for_iterator_loop<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::VertexPredecessorMap, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_populateVPMapLoopStep> __for_iterator_loop;
public:
    static CppPrimVisitor::_populateVPMapLoopStep populateVPMapLoopStep;
private:
    static read_write_property_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::VertexDescriptor, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack> __read_write_property_map1;
    static two_bit_color_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack> __two_bit_color_map;
public:
    static CppPrimVisitor::_vertexIterUnpack vertexIterUnpack;
private:
    static incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex> __incidence_and_vertex_list_and_edge_list_graph;
public:
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::OutEdgeIterator OutEdgeIterator;
    typedef pair<CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::OutEdgeIterator>::Pair OutEdgeIteratorRange;
private:
    static pair<CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::OutEdgeIterator> __pair;
public:
    struct _iterRangeBegin {
        inline CppPrimVisitor::OutEdgeIterator operator()(const CppPrimVisitor::OutEdgeIteratorRange& p) {
            return __pair.first(p);
        };
    };

    static CppPrimVisitor::_iterRangeBegin iterRangeBegin;
    struct _iterRangeEnd {
        inline CppPrimVisitor::OutEdgeIterator operator()(const CppPrimVisitor::OutEdgeIteratorRange& p) {
            return __pair.second(p);
        };
    };

    static CppPrimVisitor::_iterRangeEnd iterRangeEnd;
    struct _makeOutEdgeIteratorRange {
        inline CppPrimVisitor::OutEdgeIteratorRange operator()(const CppPrimVisitor::OutEdgeIterator& a, const CppPrimVisitor::OutEdgeIterator& b) {
            return __pair.makePair(a, b);
        };
    };

    static CppPrimVisitor::_makeOutEdgeIteratorRange makeOutEdgeIteratorRange;
    struct _outEdgeIterEnd {
        inline bool operator()(const CppPrimVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterEnd(ei);
        };
    };

    static CppPrimVisitor::_outEdgeIterEnd outEdgeIterEnd;
    struct _outEdgeIterNext {
        inline void operator()(CppPrimVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterNext(ei);
        };
    };

    static CppPrimVisitor::_outEdgeIterNext outEdgeIterNext;
    typedef base_types::Int Int;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::Graph Graph;
    struct _numVertices {
        inline CppPrimVisitor::VertexCount operator()(const CppPrimVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.numVertices(g);
        };
    };

    static CppPrimVisitor::_numVertices numVertices;
    struct _outDegree {
        inline CppPrimVisitor::VertexCount operator()(const CppPrimVisitor::VertexDescriptor& v, const CppPrimVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.outDegree(v, g);
        };
    };

    static CppPrimVisitor::_outDegree outDegree;
    struct _outEdges {
        inline void operator()(const CppPrimVisitor::VertexDescriptor& v, const CppPrimVisitor::Graph& g, CppPrimVisitor::OutEdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdges(v, g, itr);
        };
    };

    static CppPrimVisitor::_outEdges outEdges;
    struct _toVertexDescriptor {
        inline CppPrimVisitor::VertexDescriptor operator()(const CppPrimVisitor::Vertex& v, const CppPrimVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toVertexDescriptor(v, g);
        };
    };

    static CppPrimVisitor::_toVertexDescriptor toVertexDescriptor;
    struct _vertices {
        inline void operator()(const CppPrimVisitor::Graph& g, CppPrimVisitor::VertexIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.vertices(g, itr);
        };
    };

    static CppPrimVisitor::_vertices vertices;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::EdgeIterator EdgeIterator;
    struct _edgeIterEnd {
        inline bool operator()(const CppPrimVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterEnd(ei);
        };
    };

    static CppPrimVisitor::_edgeIterEnd edgeIterEnd;
    struct _edgeIterNext {
        inline void operator()(CppPrimVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterNext(ei);
        };
    };

    static CppPrimVisitor::_edgeIterNext edgeIterNext;
    struct _edges {
        inline void operator()(const CppPrimVisitor::Graph& g, CppPrimVisitor::EdgeIterator& itr) {
            return __incidence_and_vertex_list_and_edge_list_graph.edges(g, itr);
        };
    };

    static CppPrimVisitor::_edges edges;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::EdgeDescriptor EdgeDescriptor;
    struct _edgeIterUnpack {
        inline CppPrimVisitor::EdgeDescriptor operator()(const CppPrimVisitor::EdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.edgeIterUnpack(ei);
        };
    };

    static CppPrimVisitor::_edgeIterUnpack edgeIterUnpack;
    struct _outEdgeIterUnpack {
        inline CppPrimVisitor::EdgeDescriptor operator()(const CppPrimVisitor::OutEdgeIterator& ei) {
            return __incidence_and_vertex_list_and_edge_list_graph.outEdgeIterUnpack(ei);
        };
    };

    static CppPrimVisitor::_outEdgeIterUnpack outEdgeIterUnpack;
    struct _src {
        inline CppPrimVisitor::VertexDescriptor operator()(const CppPrimVisitor::EdgeDescriptor& e, const CppPrimVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.src(e, g);
        };
    };

    static CppPrimVisitor::_src src;
    struct _tgt {
        inline CppPrimVisitor::VertexDescriptor operator()(const CppPrimVisitor::EdgeDescriptor& e, const CppPrimVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.tgt(e, g);
        };
    };

    static CppPrimVisitor::_tgt tgt;
    struct _toEdgeDescriptor {
        inline CppPrimVisitor::EdgeDescriptor operator()(const CppPrimVisitor::VertexDescriptor& v1, const CppPrimVisitor::VertexDescriptor& v2, const CppPrimVisitor::Graph& g) {
            return __incidence_and_vertex_list_and_edge_list_graph.toEdgeDescriptor(v1, v2, g);
        };
    };

    static CppPrimVisitor::_toEdgeDescriptor toEdgeDescriptor;
    typedef incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex>::Edge Edge;
    struct _makeEdge {
        inline CppPrimVisitor::Edge operator()(const CppPrimVisitor::Vertex& s, const CppPrimVisitor::Vertex& t) {
            return __incidence_and_vertex_list_and_edge_list_graph.makeEdge(s, t);
        };
    };

    static CppPrimVisitor::_makeEdge makeEdge;
    typedef base_float_ops::Float Cost;
    typedef read_write_property_map<CppPrimVisitor::EdgeDescriptor, CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::Cost, CppPrimVisitor::_outEdgeIterEnd, CppPrimVisitor::_outEdgeIterNext, CppPrimVisitor::_outEdgeIterUnpack>::PropertyMap EdgeCostMap;
    struct _emptyECMap {
        inline CppPrimVisitor::EdgeCostMap operator()() {
            return __read_write_property_map.emptyMap();
        };
    };

    static CppPrimVisitor::_emptyECMap emptyECMap;
    typedef read_write_property_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::Cost, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack>::PropertyMap VertexCostMap;
    typedef triplet<CppPrimVisitor::VertexCostMap, CppPrimVisitor::VertexPredecessorMap, CppPrimVisitor::EdgeCostMap>::Triplet StateWithMaps;
    struct _getEdgeCostMap {
        inline CppPrimVisitor::EdgeCostMap operator()(const CppPrimVisitor::StateWithMaps& p) {
            return __triplet.third(p);
        };
    };

    static CppPrimVisitor::_getEdgeCostMap getEdgeCostMap;
    struct _getVertexPredecessorMap {
        inline CppPrimVisitor::VertexPredecessorMap operator()(const CppPrimVisitor::StateWithMaps& p) {
            return __triplet.second(p);
        };
    };

    static CppPrimVisitor::_getVertexPredecessorMap getVertexPredecessorMap;
    struct _putVertexPredecessorMap {
        inline CppPrimVisitor::StateWithMaps operator()(const CppPrimVisitor::VertexPredecessorMap& vpm, const CppPrimVisitor::StateWithMaps& swm) {
            return CppPrimVisitor::makeStateWithMaps(CppPrimVisitor::getVertexCostMap(swm), vpm, CppPrimVisitor::getEdgeCostMap(swm));
        };
    };

    static CppPrimVisitor::_putVertexPredecessorMap putVertexPredecessorMap;
private:
    static triplet<CppPrimVisitor::VertexCostMap, CppPrimVisitor::VertexPredecessorMap, CppPrimVisitor::EdgeCostMap> __triplet;
public:
    struct _emptyVCMap {
        inline CppPrimVisitor::VertexCostMap operator()() {
            return __read_write_property_map0.emptyMap();
        };
    };

    static CppPrimVisitor::_emptyVCMap emptyVCMap;
    struct _getVertexCostMap {
        inline CppPrimVisitor::VertexCostMap operator()(const CppPrimVisitor::StateWithMaps& p) {
            return __triplet.first(p);
        };
    };

    static CppPrimVisitor::_getVertexCostMap getVertexCostMap;
    struct _makeStateWithMaps {
        inline CppPrimVisitor::StateWithMaps operator()(const CppPrimVisitor::VertexCostMap& a, const CppPrimVisitor::VertexPredecessorMap& b, const CppPrimVisitor::EdgeCostMap& c) {
            return __triplet.makeTriplet(a, b, c);
        };
    };

    static CppPrimVisitor::_makeStateWithMaps makeStateWithMaps;
    struct _putVertexCostMap {
        inline CppPrimVisitor::StateWithMaps operator()(const CppPrimVisitor::VertexCostMap& vcm, const CppPrimVisitor::StateWithMaps& swm) {
            return CppPrimVisitor::makeStateWithMaps(vcm, CppPrimVisitor::getVertexPredecessorMap(swm), CppPrimVisitor::getEdgeCostMap(swm));
        };
    };

    static CppPrimVisitor::_putVertexCostMap putVertexCostMap;
    struct _relax {
        inline void operator()(const CppPrimVisitor::EdgeDescriptor& e, const CppPrimVisitor::Graph& g, const CppPrimVisitor::EdgeCostMap& ecm, CppPrimVisitor::VertexCostMap& vcm, CppPrimVisitor::VertexPredecessorMap& vpm) {
            CppPrimVisitor::VertexDescriptor u = CppPrimVisitor::src(e, g);
            CppPrimVisitor::VertexDescriptor v = CppPrimVisitor::tgt(e, g);
            CppPrimVisitor::Cost uCost = CppPrimVisitor::get(vcm, u);
            CppPrimVisitor::Cost vCost = CppPrimVisitor::get(vcm, v);
            CppPrimVisitor::Cost edgeCost = CppPrimVisitor::get(ecm, e);
            if (CppPrimVisitor::less(CppPrimVisitor::second(uCost, edgeCost), vCost))
            {
                CppPrimVisitor::put(vcm, v, CppPrimVisitor::second(uCost, edgeCost));
                CppPrimVisitor::put(vpm, v, u);
            }
            else
                ;
        };
    };

    static CppPrimVisitor::_relax relax;
private:
    static read_write_property_map<CppPrimVisitor::EdgeDescriptor, CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::Cost, CppPrimVisitor::_outEdgeIterEnd, CppPrimVisitor::_outEdgeIterNext, CppPrimVisitor::_outEdgeIterUnpack> __read_write_property_map;
    static read_write_property_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::Cost, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack> __read_write_property_map0;
public:
    struct _less {
        inline bool operator()(const CppPrimVisitor::Cost& i1, const CppPrimVisitor::Cost& i2) {
            return __base_float_ops.less(i1, i2);
        };
    };

    static CppPrimVisitor::_less less;
    struct _plus {
        inline CppPrimVisitor::Cost operator()(const CppPrimVisitor::Cost& i1, const CppPrimVisitor::Cost& i2) {
            return __base_float_ops.plus(i1, i2);
        };
    };

    static CppPrimVisitor::_plus plus;
    struct _primMinimumSpanningTree {
        inline void operator()(const CppPrimVisitor::Graph& g, const CppPrimVisitor::VertexDescriptor& start, CppPrimVisitor::VertexCostMap& vcm, const CppPrimVisitor::EdgeCostMap& ecm, const CppPrimVisitor::Cost& initialCost, CppPrimVisitor::VertexPredecessorMap& vpm) {
            CppPrimVisitor::put(vcm, start, initialCost);
            CppPrimVisitor::VertexIterator vertexItr;
            CppPrimVisitor::vertices(g, vertexItr);
            vpm = CppPrimVisitor::emptyVPMap();
            CppPrimVisitor::populateVPMapLoopRepeat(vertexItr, vpm, start);
            CppPrimVisitor::PriorityQueue pq = CppPrimVisitor::emptyPriorityQueue(vcm);
            CppPrimVisitor::StateWithMaps swm = CppPrimVisitor::makeStateWithMaps(vcm, vpm, ecm);
            CppPrimVisitor::ColorPropertyMap c = CppPrimVisitor::initMap(vertexItr, CppPrimVisitor::white());
            CppPrimVisitor::breadthFirstVisit(g, start, swm, pq, c);
            vcm = CppPrimVisitor::getVertexCostMap(swm);
            vpm = CppPrimVisitor::getVertexPredecessorMap(swm);
        };
    };

    static CppPrimVisitor::_primMinimumSpanningTree primMinimumSpanningTree;
    struct _second {
        inline CppPrimVisitor::Cost operator()(const CppPrimVisitor::Cost& c1, const CppPrimVisitor::Cost& c2) {
            return c2;
        };
    };

    static CppPrimVisitor::_second second;
    typedef two_bit_color_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack>::ColorPropertyMap ColorPropertyMap;
    typedef two_bit_color_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack>::Color Color;
    struct _black {
        inline CppPrimVisitor::Color operator()() {
            return __two_bit_color_map.black();
        };
    };

    static CppPrimVisitor::_black black;
    struct _get {
        inline CppPrimVisitor::Cost operator()(const CppPrimVisitor::VertexCostMap& pm, const CppPrimVisitor::VertexDescriptor& k) {
            return __read_write_property_map0.get(pm, k);
        };
        inline CppPrimVisitor::VertexDescriptor operator()(const CppPrimVisitor::VertexPredecessorMap& pm, const CppPrimVisitor::VertexDescriptor& k) {
            return __read_write_property_map1.get(pm, k);
        };
        inline CppPrimVisitor::Cost operator()(const CppPrimVisitor::EdgeCostMap& pm, const CppPrimVisitor::EdgeDescriptor& k) {
            return __read_write_property_map.get(pm, k);
        };
        inline CppPrimVisitor::Color operator()(const CppPrimVisitor::ColorPropertyMap& pm, const CppPrimVisitor::VertexDescriptor& k) {
            return __two_bit_color_map.get(pm, k);
        };
    };

    typedef priority_queue<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::Cost, CppPrimVisitor::VertexCostMap, CppPrimVisitor::_get>::PriorityQueue PriorityQueue;
    struct _bfsInnerLoopRepeat {
        inline void operator()(const CppPrimVisitor::OutEdgeIterator& itr, CppPrimVisitor::StateWithMaps& s1, CppPrimVisitor::PriorityQueue& s2, CppPrimVisitor::ColorPropertyMap& s3, const CppPrimVisitor::Graph& ctx1, const CppPrimVisitor::VertexDescriptor& ctx2) {
            return __for_iterator_loop3_2.forLoopRepeat(itr, s1, s2, s3, ctx1, ctx2);
        };
    };

    static CppPrimVisitor::_bfsInnerLoopRepeat bfsInnerLoopRepeat;
    struct _bfsInnerLoopStep {
        inline void operator()(const CppPrimVisitor::OutEdgeIterator& edgeItr, CppPrimVisitor::StateWithMaps& x, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::ColorPropertyMap& c, const CppPrimVisitor::Graph& g, const CppPrimVisitor::VertexDescriptor& u) {
            CppPrimVisitor::EdgeDescriptor e = CppPrimVisitor::outEdgeIterUnpack(edgeItr);
            CppPrimVisitor::VertexDescriptor v = CppPrimVisitor::tgt(e, g);
            CppPrimVisitor::examineEdge(e, g, q, x);
            CppPrimVisitor::Color vc = CppPrimVisitor::get(c, v);
            if ((vc) == (CppPrimVisitor::white()))
            {
                CppPrimVisitor::treeEdge(e, g, q, x);
                CppPrimVisitor::put(c, v, CppPrimVisitor::gray());
                CppPrimVisitor::discoverVertex(v, g, q, x);
                CppPrimVisitor::push(v, q);
            }
            else
                if ((vc) == (CppPrimVisitor::gray()))
                {
                    CppPrimVisitor::grayTarget(e, g, q, x);
                }
                else
                {
                    CppPrimVisitor::blackTarget(e, g, q, x);
                }
        };
    };

private:
    static for_iterator_loop3_2<CppPrimVisitor::Graph, CppPrimVisitor::VertexDescriptor, CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::StateWithMaps, CppPrimVisitor::PriorityQueue, CppPrimVisitor::ColorPropertyMap, CppPrimVisitor::_outEdgeIterEnd, CppPrimVisitor::_outEdgeIterNext, CppPrimVisitor::_bfsInnerLoopStep> __for_iterator_loop3_2;
public:
    static CppPrimVisitor::_bfsInnerLoopStep bfsInnerLoopStep;
    struct _bfsOuterLoopCond {
        inline bool operator()(const CppPrimVisitor::StateWithMaps& a, const CppPrimVisitor::PriorityQueue& q, const CppPrimVisitor::ColorPropertyMap& c, const CppPrimVisitor::Graph& g) {
            return !CppPrimVisitor::isEmptyQueue(q);
        };
    };

    static CppPrimVisitor::_bfsOuterLoopCond bfsOuterLoopCond;
    struct _bfsOuterLoopRepeat {
        inline void operator()(CppPrimVisitor::StateWithMaps& s1, CppPrimVisitor::PriorityQueue& s2, CppPrimVisitor::ColorPropertyMap& s3, const CppPrimVisitor::Graph& ctx) {
            return __while_loop3.repeat(s1, s2, s3, ctx);
        };
    };

    static CppPrimVisitor::_bfsOuterLoopRepeat bfsOuterLoopRepeat;
    struct _bfsOuterLoopStep {
        inline void operator()(CppPrimVisitor::StateWithMaps& x, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::ColorPropertyMap& c, const CppPrimVisitor::Graph& g) {
            CppPrimVisitor::VertexDescriptor u = CppPrimVisitor::front(q);
            CppPrimVisitor::pop(q);
            CppPrimVisitor::examineVertex(u, g, q, x);
            CppPrimVisitor::OutEdgeIterator edgeItr;
            CppPrimVisitor::outEdges(u, g, edgeItr);
            CppPrimVisitor::bfsInnerLoopRepeat(edgeItr, x, q, c, g, u);
            CppPrimVisitor::put(c, u, CppPrimVisitor::black());
            CppPrimVisitor::finishVertex(u, g, q, x);
        };
    };

private:
    static while_loop3<CppPrimVisitor::Graph, CppPrimVisitor::StateWithMaps, CppPrimVisitor::PriorityQueue, CppPrimVisitor::ColorPropertyMap, CppPrimVisitor::_bfsOuterLoopCond, CppPrimVisitor::_bfsOuterLoopStep> __while_loop3;
public:
    static CppPrimVisitor::_bfsOuterLoopStep bfsOuterLoopStep;
    struct _blackTarget {
        inline void operator()(const CppPrimVisitor::EdgeDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_blackTarget blackTarget;
    struct _breadthFirstVisit {
        inline void operator()(const CppPrimVisitor::Graph& g, const CppPrimVisitor::VertexDescriptor& s, CppPrimVisitor::StateWithMaps& a, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::ColorPropertyMap& c) {
            CppPrimVisitor::discoverVertex(s, g, q, a);
            CppPrimVisitor::push(s, q);
            CppPrimVisitor::put(c, s, CppPrimVisitor::gray());
            CppPrimVisitor::bfsOuterLoopRepeat(a, q, c, g);
        };
    };

    static CppPrimVisitor::_breadthFirstVisit breadthFirstVisit;
    struct _discoverVertex {
        inline void operator()(const CppPrimVisitor::VertexDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_discoverVertex discoverVertex;
    struct _emptyPriorityQueue {
        inline CppPrimVisitor::PriorityQueue operator()(const CppPrimVisitor::VertexCostMap& pm) {
            return __priority_queue.empty(pm);
        };
    };

    static CppPrimVisitor::_emptyPriorityQueue emptyPriorityQueue;
    struct _examineEdge {
        inline void operator()(const CppPrimVisitor::EdgeDescriptor& e, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& pq, CppPrimVisitor::StateWithMaps& swm) {
            CppPrimVisitor::VertexCostMap origVcm = CppPrimVisitor::getVertexCostMap(swm);
            CppPrimVisitor::VertexPredecessorMap vpm = CppPrimVisitor::getVertexPredecessorMap(swm);
            CppPrimVisitor::EdgeCostMap ecm = CppPrimVisitor::getEdgeCostMap(swm);
            CppPrimVisitor::VertexCostMap vcm = origVcm;
            CppPrimVisitor::relax(e, g, ecm, vcm, vpm);
            if ((vcm) == (origVcm))
                ;
            else
            {
                swm = CppPrimVisitor::putVertexPredecessorMap(vpm, CppPrimVisitor::putVertexCostMap(vcm, swm));
                pq = CppPrimVisitor::update(vcm, CppPrimVisitor::tgt(e, g), pq);
            }
        };
    };

    static CppPrimVisitor::_examineEdge examineEdge;
    struct _examineVertex {
        inline void operator()(const CppPrimVisitor::VertexDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_examineVertex examineVertex;
    struct _finishVertex {
        inline void operator()(const CppPrimVisitor::VertexDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_finishVertex finishVertex;
    struct _front {
        inline CppPrimVisitor::VertexDescriptor operator()(const CppPrimVisitor::PriorityQueue& q) {
            return __priority_queue.front(q);
        };
    };

    static CppPrimVisitor::_front front;
    struct _grayTarget {
        inline void operator()(const CppPrimVisitor::EdgeDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_grayTarget grayTarget;
    struct _isEmptyQueue {
        inline bool operator()(const CppPrimVisitor::PriorityQueue& q) {
            return __priority_queue.isEmpty(q);
        };
    };

    static CppPrimVisitor::_isEmptyQueue isEmptyQueue;
    struct _nonTreeEdge {
        inline void operator()(const CppPrimVisitor::EdgeDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_nonTreeEdge nonTreeEdge;
    struct _pop {
        inline void operator()(CppPrimVisitor::PriorityQueue& q) {
            return __priority_queue.pop(q);
        };
    };

    static CppPrimVisitor::_pop pop;
    struct _push {
        inline void operator()(const CppPrimVisitor::VertexDescriptor& a, CppPrimVisitor::PriorityQueue& q) {
            return __priority_queue.push(a, q);
        };
    };

    static CppPrimVisitor::_push push;
    struct _treeEdge {
        inline void operator()(const CppPrimVisitor::EdgeDescriptor& edgeOrVertex, const CppPrimVisitor::Graph& g, CppPrimVisitor::PriorityQueue& q, CppPrimVisitor::StateWithMaps& a) {
            ;
        };
    };

    static CppPrimVisitor::_treeEdge treeEdge;
    struct _update {
        inline CppPrimVisitor::PriorityQueue operator()(const CppPrimVisitor::VertexCostMap& pm, const CppPrimVisitor::VertexDescriptor& a, const CppPrimVisitor::PriorityQueue& pq) {
            return __priority_queue.update(pm, a, pq);
        };
    };

    static CppPrimVisitor::_update update;
private:
    static priority_queue<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::Cost, CppPrimVisitor::VertexCostMap, CppPrimVisitor::_get> __priority_queue;
public:
    static CppPrimVisitor::_get get;
    struct _gray {
        inline CppPrimVisitor::Color operator()() {
            return __two_bit_color_map.gray();
        };
    };

    static CppPrimVisitor::_gray gray;
    struct _initMap {
        inline CppPrimVisitor::VertexCostMap operator()(const CppPrimVisitor::VertexIterator& kli, const CppPrimVisitor::Cost& v) {
            return __read_write_property_map0.initMap(kli, v);
        };
        inline CppPrimVisitor::VertexPredecessorMap operator()(const CppPrimVisitor::VertexIterator& kli, const CppPrimVisitor::VertexDescriptor& v) {
            return __read_write_property_map1.initMap(kli, v);
        };
        inline CppPrimVisitor::EdgeCostMap operator()(const CppPrimVisitor::OutEdgeIterator& kli, const CppPrimVisitor::Cost& v) {
            return __read_write_property_map.initMap(kli, v);
        };
        inline CppPrimVisitor::ColorPropertyMap operator()(const CppPrimVisitor::VertexIterator& kli, const CppPrimVisitor::Color& v) {
            return __two_bit_color_map.initMap(kli, v);
        };
    };

    static CppPrimVisitor::_initMap initMap;
    struct _put {
        inline void operator()(CppPrimVisitor::VertexCostMap& pm, const CppPrimVisitor::VertexDescriptor& k, const CppPrimVisitor::Cost& v) {
            return __read_write_property_map0.put(pm, k, v);
        };
        inline void operator()(CppPrimVisitor::VertexPredecessorMap& pm, const CppPrimVisitor::VertexDescriptor& k, const CppPrimVisitor::VertexDescriptor& v) {
            return __read_write_property_map1.put(pm, k, v);
        };
        inline void operator()(CppPrimVisitor::EdgeCostMap& pm, const CppPrimVisitor::EdgeDescriptor& k, const CppPrimVisitor::Cost& v) {
            return __read_write_property_map.put(pm, k, v);
        };
        inline void operator()(CppPrimVisitor::ColorPropertyMap& pm, const CppPrimVisitor::VertexDescriptor& k, const CppPrimVisitor::Color& v) {
            return __two_bit_color_map.put(pm, k, v);
        };
    };

    static CppPrimVisitor::_put put;
    struct _white {
        inline CppPrimVisitor::Color operator()() {
            return __two_bit_color_map.white();
        };
    };

    static CppPrimVisitor::_white white;
};
} // examples
} // bgl
} // mg_src
} // bgl_cpp