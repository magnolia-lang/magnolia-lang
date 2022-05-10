#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"


namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_types CppBFSTestVisitor::__base_types;



    CppBFSTestVisitor::_emptyVertexVector CppBFSTestVisitor::emptyVertexVector;

    fifo_queue<CppBFSTestVisitor::VertexDescriptor> CppBFSTestVisitor::__fifo_queue;

    vector<CppBFSTestVisitor::VertexDescriptor> CppBFSTestVisitor::__vector;



    CppBFSTestVisitor::_pushBack CppBFSTestVisitor::pushBack;



    CppBFSTestVisitor::_vertexIterEnd CppBFSTestVisitor::vertexIterEnd;



    CppBFSTestVisitor::_vertexIterNext CppBFSTestVisitor::vertexIterNext;



    two_bit_color_map<CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::VertexIterator, CppBFSTestVisitor::_vertexIterEnd, CppBFSTestVisitor::_vertexIterNext, CppBFSTestVisitor::_vertexIterUnpack> CppBFSTestVisitor::__two_bit_color_map;

    CppBFSTestVisitor::_vertexIterUnpack CppBFSTestVisitor::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppBFSTestVisitor::Vertex> CppBFSTestVisitor::__incidence_and_vertex_list_and_edge_list_graph;

    pair<CppBFSTestVisitor::OutEdgeIterator, CppBFSTestVisitor::OutEdgeIterator> CppBFSTestVisitor::__pair;



    CppBFSTestVisitor::_iterRangeBegin CppBFSTestVisitor::iterRangeBegin;



    CppBFSTestVisitor::_iterRangeEnd CppBFSTestVisitor::iterRangeEnd;



    CppBFSTestVisitor::_makeOutEdgeIteratorRange CppBFSTestVisitor::makeOutEdgeIteratorRange;



    CppBFSTestVisitor::_outEdgeIterEnd CppBFSTestVisitor::outEdgeIterEnd;



    CppBFSTestVisitor::_outEdgeIterNext CppBFSTestVisitor::outEdgeIterNext;



    CppBFSTestVisitor::_breadthFirstSearch CppBFSTestVisitor::breadthFirstSearch;



    CppBFSTestVisitor::_numVertices CppBFSTestVisitor::numVertices;



    CppBFSTestVisitor::_outDegree CppBFSTestVisitor::outDegree;



    CppBFSTestVisitor::_outEdges CppBFSTestVisitor::outEdges;



    CppBFSTestVisitor::_toVertexDescriptor CppBFSTestVisitor::toVertexDescriptor;



    CppBFSTestVisitor::_vertices CppBFSTestVisitor::vertices;



    CppBFSTestVisitor::_empty CppBFSTestVisitor::empty;



    CppBFSTestVisitor::_examineVertex CppBFSTestVisitor::examineVertex;



    CppBFSTestVisitor::_front CppBFSTestVisitor::front;



    CppBFSTestVisitor::_isEmptyQueue CppBFSTestVisitor::isEmptyQueue;



    CppBFSTestVisitor::_pop CppBFSTestVisitor::pop;



    CppBFSTestVisitor::_push CppBFSTestVisitor::push;



    CppBFSTestVisitor::_pushPopBehavior CppBFSTestVisitor::pushPopBehavior;



    CppBFSTestVisitor::_edgeIterEnd CppBFSTestVisitor::edgeIterEnd;



    CppBFSTestVisitor::_edgeIterNext CppBFSTestVisitor::edgeIterNext;



    CppBFSTestVisitor::_edges CppBFSTestVisitor::edges;



    CppBFSTestVisitor::_defaultAction CppBFSTestVisitor::defaultAction;



    CppBFSTestVisitor::_edgeIterUnpack CppBFSTestVisitor::edgeIterUnpack;



    CppBFSTestVisitor::_outEdgeIterUnpack CppBFSTestVisitor::outEdgeIterUnpack;



    CppBFSTestVisitor::_src CppBFSTestVisitor::src;



    CppBFSTestVisitor::_tgt CppBFSTestVisitor::tgt;



    CppBFSTestVisitor::_toEdgeDescriptor CppBFSTestVisitor::toEdgeDescriptor;



    CppBFSTestVisitor::_makeEdge CppBFSTestVisitor::makeEdge;



    CppBFSTestVisitor::_bfsInnerLoopRepeat CppBFSTestVisitor::bfsInnerLoopRepeat;



    for_iterator_loop3_2<CppBFSTestVisitor::Graph, CppBFSTestVisitor::VertexDescriptor, CppBFSTestVisitor::OutEdgeIterator, CppBFSTestVisitor::VertexVector, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap, CppBFSTestVisitor::_outEdgeIterEnd, CppBFSTestVisitor::_outEdgeIterNext, CppBFSTestVisitor::_bfsInnerLoopStep> CppBFSTestVisitor::__for_iterator_loop3_2;

    CppBFSTestVisitor::_bfsInnerLoopStep CppBFSTestVisitor::bfsInnerLoopStep;



    CppBFSTestVisitor::_bfsOuterLoopCond CppBFSTestVisitor::bfsOuterLoopCond;



    CppBFSTestVisitor::_bfsOuterLoopRepeat CppBFSTestVisitor::bfsOuterLoopRepeat;



    while_loop3<CppBFSTestVisitor::Graph, CppBFSTestVisitor::VertexVector, CppBFSTestVisitor::FIFOQueue, CppBFSTestVisitor::ColorPropertyMap, CppBFSTestVisitor::_bfsOuterLoopCond, CppBFSTestVisitor::_bfsOuterLoopStep> CppBFSTestVisitor::__while_loop3;

    CppBFSTestVisitor::_bfsOuterLoopStep CppBFSTestVisitor::bfsOuterLoopStep;



    CppBFSTestVisitor::_breadthFirstVisit CppBFSTestVisitor::breadthFirstVisit;



    CppBFSTestVisitor::_black CppBFSTestVisitor::black;



    CppBFSTestVisitor::_get CppBFSTestVisitor::get;



    CppBFSTestVisitor::_gray CppBFSTestVisitor::gray;



    CppBFSTestVisitor::_initMap CppBFSTestVisitor::initMap;



    CppBFSTestVisitor::_put CppBFSTestVisitor::put;



    CppBFSTestVisitor::_white CppBFSTestVisitor::white;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_unit CppBellmanFord::__base_unit;

    base_types CppBellmanFord::__base_types;

    base_float_ops CppBellmanFord::__base_float_ops;

    base_bool CppBellmanFord::__base_bool;



    CppBellmanFord::_vertexIterEnd CppBellmanFord::vertexIterEnd;



    CppBellmanFord::_vertexIterNext CppBellmanFord::vertexIterNext;





    CppBellmanFord::_emptyVPMap CppBellmanFord::emptyVPMap;

    read_write_property_map<CppBellmanFord::VertexDescriptor, CppBellmanFord::VertexIterator, CppBellmanFord::VertexDescriptor, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_vertexIterUnpack> CppBellmanFord::__read_write_property_map1;

    CppBellmanFord::_vertexIterUnpack CppBellmanFord::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppBellmanFord::Vertex> CppBellmanFord::__incidence_and_vertex_list_and_edge_list_graph;



    CppBellmanFord::_forIterationEnd CppBellmanFord::forIterationEnd;



    CppBellmanFord::_populateVPMapLoopRepeat CppBellmanFord::populateVPMapLoopRepeat;



    for_iterator_loop<CppBellmanFord::Unit, CppBellmanFord::VertexIterator, CppBellmanFord::VertexPredecessorMap, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_populateVPMapLoopStep> CppBellmanFord::__for_iterator_loop;

    CppBellmanFord::_populateVPMapLoopStep CppBellmanFord::populateVPMapLoopStep;



    CppBellmanFord::_unit CppBellmanFord::unit;



    CppBellmanFord::_outEdgeIterEnd CppBellmanFord::outEdgeIterEnd;



    CppBellmanFord::_outEdgeIterNext CppBellmanFord::outEdgeIterNext;



    CppBellmanFord::_numVertices CppBellmanFord::numVertices;



    CppBellmanFord::_outDegree CppBellmanFord::outDegree;



    CppBellmanFord::_outEdges CppBellmanFord::outEdges;



    CppBellmanFord::_toVertexDescriptor CppBellmanFord::toVertexDescriptor;



    CppBellmanFord::_vertices CppBellmanFord::vertices;



    CppBellmanFord::_edgeIterEnd CppBellmanFord::edgeIterEnd;



    CppBellmanFord::_edgeIterNext CppBellmanFord::edgeIterNext;



    CppBellmanFord::_edges CppBellmanFord::edges;



    CppBellmanFord::_edgeIterUnpack CppBellmanFord::edgeIterUnpack;



    CppBellmanFord::_edgeMinimized CppBellmanFord::edgeMinimized;



    CppBellmanFord::_edgeNotMinimized CppBellmanFord::edgeNotMinimized;



    CppBellmanFord::_edgeNotRelaxed CppBellmanFord::edgeNotRelaxed;



    CppBellmanFord::_edgeRelaxed CppBellmanFord::edgeRelaxed;



    CppBellmanFord::_examineEdge CppBellmanFord::examineEdge;



    CppBellmanFord::_outEdgeIterUnpack CppBellmanFord::outEdgeIterUnpack;



    CppBellmanFord::_src CppBellmanFord::src;



    CppBellmanFord::_tgt CppBellmanFord::tgt;



    CppBellmanFord::_toEdgeDescriptor CppBellmanFord::toEdgeDescriptor;



    CppBellmanFord::_makeEdge CppBellmanFord::makeEdge;



    CppBellmanFord::_emptyECMap CppBellmanFord::emptyECMap;



    CppBellmanFord::_edgeRelaxationInnerLoopRepeat CppBellmanFord::edgeRelaxationInnerLoopRepeat;



    for_iterator_loop3_2<CppBellmanFord::EdgeCostMap, CppBellmanFord::Graph, CppBellmanFord::EdgeIterator, CppBellmanFord::Unit, CppBellmanFord::VertexCostMap, CppBellmanFord::VertexPredecessorMap, CppBellmanFord::_edgeIterEnd, CppBellmanFord::_edgeIterNext, CppBellmanFord::_edgeRelaxationInnerLoopStep> CppBellmanFord::__for_iterator_loop3_2;

    CppBellmanFord::_edgeRelaxationInnerLoopStep CppBellmanFord::edgeRelaxationInnerLoopStep;



    CppBellmanFord::_edgeRelaxationOuterLoopRepeat CppBellmanFord::edgeRelaxationOuterLoopRepeat;



    for_iterator_loop3_2<CppBellmanFord::EdgeCostMap, CppBellmanFord::Graph, CppBellmanFord::VertexIterator, CppBellmanFord::Unit, CppBellmanFord::VertexCostMap, CppBellmanFord::VertexPredecessorMap, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_edgeRelaxationOuterLoopStep> CppBellmanFord::__for_iterator_loop3_20;

    CppBellmanFord::_edgeRelaxationOuterLoopStep CppBellmanFord::edgeRelaxationOuterLoopStep;



    CppBellmanFord::_emptyVCMap CppBellmanFord::emptyVCMap;



    CppBellmanFord::_relax CppBellmanFord::relax;

    read_write_property_map<CppBellmanFord::EdgeDescriptor, CppBellmanFord::OutEdgeIterator, CppBellmanFord::Cost, CppBellmanFord::_outEdgeIterEnd, CppBellmanFord::_outEdgeIterNext, CppBellmanFord::_outEdgeIterUnpack> CppBellmanFord::__read_write_property_map;

    read_write_property_map<CppBellmanFord::VertexDescriptor, CppBellmanFord::VertexIterator, CppBellmanFord::Cost, CppBellmanFord::_vertexIterEnd, CppBellmanFord::_vertexIterNext, CppBellmanFord::_vertexIterUnpack> CppBellmanFord::__read_write_property_map0;



    CppBellmanFord::_get CppBellmanFord::get;



    CppBellmanFord::_inf CppBellmanFord::inf;



    CppBellmanFord::_initMap CppBellmanFord::initMap;



    CppBellmanFord::_less CppBellmanFord::less;



    CppBellmanFord::_negate CppBellmanFord::negate;



    CppBellmanFord::_plus CppBellmanFord::plus;



    CppBellmanFord::_put CppBellmanFord::put;



    CppBellmanFord::_zero CppBellmanFord::zero;



    CppBellmanFord::_bellmanFordShortestPaths CppBellmanFord::bellmanFordShortestPaths;



    CppBellmanFord::_bfalse CppBellmanFord::bfalse;



    CppBellmanFord::_btrue CppBellmanFord::btrue;



    CppBellmanFord::_checkNegativeCycleLoopRepeat CppBellmanFord::checkNegativeCycleLoopRepeat;



    for_iterator_loop2_3<CppBellmanFord::VertexCostMap, CppBellmanFord::EdgeCostMap, CppBellmanFord::Graph, CppBellmanFord::EdgeIterator, CppBellmanFord::Unit, CppBellmanFord::Bool, CppBellmanFord::_edgeIterEnd, CppBellmanFord::_edgeIterNext, CppBellmanFord::_checkNegativeCycleLoopStep> CppBellmanFord::__for_iterator_loop2_3;

    CppBellmanFord::_checkNegativeCycleLoopStep CppBellmanFord::checkNegativeCycleLoopStep;



    CppBellmanFord::_holds CppBellmanFord::holds;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_types CppCustomGraphTypeBFSTestVisitor::__base_types;



    CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd CppCustomGraphTypeBFSTestVisitor::vertexIterEnd;



    CppCustomGraphTypeBFSTestVisitor::_vertexIterNext CppCustomGraphTypeBFSTestVisitor::vertexIterNext;



    CppCustomGraphTypeBFSTestVisitor::_emptyVertexList CppCustomGraphTypeBFSTestVisitor::emptyVertexList;



    CppCustomGraphTypeBFSTestVisitor::_getVertexIterator CppCustomGraphTypeBFSTestVisitor::getVertexIterator;



    CppCustomGraphTypeBFSTestVisitor::_emptyVertexVector CppCustomGraphTypeBFSTestVisitor::emptyVertexVector;

    edge_without_descriptor<CppCustomGraphTypeBFSTestVisitor::Vertex> CppCustomGraphTypeBFSTestVisitor::__edge_without_descriptor;

    fifo_queue<CppCustomGraphTypeBFSTestVisitor::Vertex> CppCustomGraphTypeBFSTestVisitor::__fifo_queue;

    iterable_list<CppCustomGraphTypeBFSTestVisitor::Vertex> CppCustomGraphTypeBFSTestVisitor::__iterable_list0;

    vector<CppCustomGraphTypeBFSTestVisitor::Vertex> CppCustomGraphTypeBFSTestVisitor::__vector;



    CppCustomGraphTypeBFSTestVisitor::_pushBack CppCustomGraphTypeBFSTestVisitor::pushBack;



    two_bit_color_map<CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack> CppCustomGraphTypeBFSTestVisitor::__two_bit_color_map;

    CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack CppCustomGraphTypeBFSTestVisitor::vertexIterUnpack;



    CppCustomGraphTypeBFSTestVisitor::_empty CppCustomGraphTypeBFSTestVisitor::empty;



    CppCustomGraphTypeBFSTestVisitor::_front CppCustomGraphTypeBFSTestVisitor::front;



    CppCustomGraphTypeBFSTestVisitor::_isEmptyQueue CppCustomGraphTypeBFSTestVisitor::isEmptyQueue;



    CppCustomGraphTypeBFSTestVisitor::_pop CppCustomGraphTypeBFSTestVisitor::pop;



    CppCustomGraphTypeBFSTestVisitor::_push CppCustomGraphTypeBFSTestVisitor::push;



    CppCustomGraphTypeBFSTestVisitor::_pushPopBehavior CppCustomGraphTypeBFSTestVisitor::pushPopBehavior;



    CppCustomGraphTypeBFSTestVisitor::_edgeIterEnd CppCustomGraphTypeBFSTestVisitor::edgeIterEnd;



    CppCustomGraphTypeBFSTestVisitor::_edgeIterNext CppCustomGraphTypeBFSTestVisitor::edgeIterNext;



    CppCustomGraphTypeBFSTestVisitor::_emptyEdgeList CppCustomGraphTypeBFSTestVisitor::emptyEdgeList;



    CppCustomGraphTypeBFSTestVisitor::_getEdgeIterator CppCustomGraphTypeBFSTestVisitor::getEdgeIterator;



    CppCustomGraphTypeBFSTestVisitor::_isEmpty CppCustomGraphTypeBFSTestVisitor::isEmpty;



    CppCustomGraphTypeBFSTestVisitor::_tail CppCustomGraphTypeBFSTestVisitor::tail;

    pair<CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator> CppCustomGraphTypeBFSTestVisitor::__pair;



    CppCustomGraphTypeBFSTestVisitor::_getOutEdgeIterator CppCustomGraphTypeBFSTestVisitor::getOutEdgeIterator;



    CppCustomGraphTypeBFSTestVisitor::_iterRangeBegin CppCustomGraphTypeBFSTestVisitor::iterRangeBegin;



    CppCustomGraphTypeBFSTestVisitor::_iterRangeEnd CppCustomGraphTypeBFSTestVisitor::iterRangeEnd;



    CppCustomGraphTypeBFSTestVisitor::_makeOutEdgeIteratorRange CppCustomGraphTypeBFSTestVisitor::makeOutEdgeIteratorRange;



    CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd CppCustomGraphTypeBFSTestVisitor::outEdgeIterEnd;



    CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext CppCustomGraphTypeBFSTestVisitor::outEdgeIterNext;

    iterable_list<CppCustomGraphTypeBFSTestVisitor::Edge> CppCustomGraphTypeBFSTestVisitor::__iterable_list;



    CppCustomGraphTypeBFSTestVisitor::_cons CppCustomGraphTypeBFSTestVisitor::cons;



    CppCustomGraphTypeBFSTestVisitor::_edgeIterUnpack CppCustomGraphTypeBFSTestVisitor::edgeIterUnpack;



    CppCustomGraphTypeBFSTestVisitor::_head CppCustomGraphTypeBFSTestVisitor::head;



    CppCustomGraphTypeBFSTestVisitor::_makeEdge CppCustomGraphTypeBFSTestVisitor::makeEdge;



    CppCustomGraphTypeBFSTestVisitor::_outEdgeIterUnpack CppCustomGraphTypeBFSTestVisitor::outEdgeIterUnpack;



    CppCustomGraphTypeBFSTestVisitor::_srcPlainEdge CppCustomGraphTypeBFSTestVisitor::srcPlainEdge;





    CppCustomGraphTypeBFSTestVisitor::_breadthFirstSearch CppCustomGraphTypeBFSTestVisitor::breadthFirstSearch;



    CppCustomGraphTypeBFSTestVisitor::_defaultAction CppCustomGraphTypeBFSTestVisitor::defaultAction;



    CppCustomGraphTypeBFSTestVisitor::_edges CppCustomGraphTypeBFSTestVisitor::edges;



    CppCustomGraphTypeBFSTestVisitor::_examineVertex CppCustomGraphTypeBFSTestVisitor::examineVertex;



    CppCustomGraphTypeBFSTestVisitor::_outEdges CppCustomGraphTypeBFSTestVisitor::outEdges;



    CppCustomGraphTypeBFSTestVisitor::_src CppCustomGraphTypeBFSTestVisitor::src;



    CppCustomGraphTypeBFSTestVisitor::_tgt CppCustomGraphTypeBFSTestVisitor::tgt;



    CppCustomGraphTypeBFSTestVisitor::_vertices CppCustomGraphTypeBFSTestVisitor::vertices;



    CppCustomGraphTypeBFSTestVisitor::_numVertices CppCustomGraphTypeBFSTestVisitor::numVertices;



    CppCustomGraphTypeBFSTestVisitor::_outDegree CppCustomGraphTypeBFSTestVisitor::outDegree;

    custom_incidence_and_vertex_list_and_edge_list_graph<CppCustomGraphTypeBFSTestVisitor::Edge, CppCustomGraphTypeBFSTestVisitor::EdgeIterator, CppCustomGraphTypeBFSTestVisitor::EdgeList, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::VertexIterator, CppCustomGraphTypeBFSTestVisitor::VertexList, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_cons, CppCustomGraphTypeBFSTestVisitor::_edgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_edgeIterNext, CppCustomGraphTypeBFSTestVisitor::_edgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_emptyEdgeList, CppCustomGraphTypeBFSTestVisitor::_emptyVertexList, CppCustomGraphTypeBFSTestVisitor::_getEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getOutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::_getVertexIterator, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_head, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_isEmpty, CppCustomGraphTypeBFSTestVisitor::_makeEdge, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterUnpack, CppCustomGraphTypeBFSTestVisitor::_srcPlainEdge, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tail, CppCustomGraphTypeBFSTestVisitor::_tgtPlainEdge, CppCustomGraphTypeBFSTestVisitor::_vertexIterEnd, CppCustomGraphTypeBFSTestVisitor::_vertexIterNext, CppCustomGraphTypeBFSTestVisitor::_vertexIterUnpack> CppCustomGraphTypeBFSTestVisitor::__custom_incidence_and_vertex_list_and_edge_list_graph;

    CppCustomGraphTypeBFSTestVisitor::_tgtPlainEdge CppCustomGraphTypeBFSTestVisitor::tgtPlainEdge;



    CppCustomGraphTypeBFSTestVisitor::_bfsInnerLoopRepeat CppCustomGraphTypeBFSTestVisitor::bfsInnerLoopRepeat;



    for_iterator_loop3_2<CppCustomGraphTypeBFSTestVisitor::Graph, CppCustomGraphTypeBFSTestVisitor::Vertex, CppCustomGraphTypeBFSTestVisitor::OutEdgeIterator, CppCustomGraphTypeBFSTestVisitor::VertexVector, CppCustomGraphTypeBFSTestVisitor::FIFOQueue, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterEnd, CppCustomGraphTypeBFSTestVisitor::_outEdgeIterNext, CppCustomGraphTypeBFSTestVisitor::_bfsInnerLoopStep> CppCustomGraphTypeBFSTestVisitor::__for_iterator_loop3_2;

    CppCustomGraphTypeBFSTestVisitor::_bfsInnerLoopStep CppCustomGraphTypeBFSTestVisitor::bfsInnerLoopStep;



    CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopCond CppCustomGraphTypeBFSTestVisitor::bfsOuterLoopCond;



    CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopRepeat CppCustomGraphTypeBFSTestVisitor::bfsOuterLoopRepeat;



    while_loop3<CppCustomGraphTypeBFSTestVisitor::Graph, CppCustomGraphTypeBFSTestVisitor::VertexVector, CppCustomGraphTypeBFSTestVisitor::FIFOQueue, CppCustomGraphTypeBFSTestVisitor::ColorPropertyMap, CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopCond, CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopStep> CppCustomGraphTypeBFSTestVisitor::__while_loop3;

    CppCustomGraphTypeBFSTestVisitor::_bfsOuterLoopStep CppCustomGraphTypeBFSTestVisitor::bfsOuterLoopStep;



    CppCustomGraphTypeBFSTestVisitor::_breadthFirstVisit CppCustomGraphTypeBFSTestVisitor::breadthFirstVisit;



    CppCustomGraphTypeBFSTestVisitor::_black CppCustomGraphTypeBFSTestVisitor::black;



    CppCustomGraphTypeBFSTestVisitor::_get CppCustomGraphTypeBFSTestVisitor::get;



    CppCustomGraphTypeBFSTestVisitor::_gray CppCustomGraphTypeBFSTestVisitor::gray;



    CppCustomGraphTypeBFSTestVisitor::_initMap CppCustomGraphTypeBFSTestVisitor::initMap;



    CppCustomGraphTypeBFSTestVisitor::_put CppCustomGraphTypeBFSTestVisitor::put;



    CppCustomGraphTypeBFSTestVisitor::_white CppCustomGraphTypeBFSTestVisitor::white;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {


    CppDFSTestVisitor::_emptyStackIsEmpty CppDFSTestVisitor::emptyStackIsEmpty;

    base_types CppDFSTestVisitor::__base_types;



    CppDFSTestVisitor::_emptyVertexVector CppDFSTestVisitor::emptyVertexVector;

    stack<CppDFSTestVisitor::VertexDescriptor> CppDFSTestVisitor::__stack;

    vector<CppDFSTestVisitor::VertexDescriptor> CppDFSTestVisitor::__vector;



    CppDFSTestVisitor::_pushBack CppDFSTestVisitor::pushBack;



    CppDFSTestVisitor::_vertexIterEnd CppDFSTestVisitor::vertexIterEnd;



    CppDFSTestVisitor::_vertexIterNext CppDFSTestVisitor::vertexIterNext;



    two_bit_color_map<CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::VertexIterator, CppDFSTestVisitor::_vertexIterEnd, CppDFSTestVisitor::_vertexIterNext, CppDFSTestVisitor::_vertexIterUnpack> CppDFSTestVisitor::__two_bit_color_map;

    CppDFSTestVisitor::_vertexIterUnpack CppDFSTestVisitor::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppDFSTestVisitor::Vertex> CppDFSTestVisitor::__incidence_and_vertex_list_and_edge_list_graph;



    CppDFSTestVisitor::_empty CppDFSTestVisitor::empty;



    CppDFSTestVisitor::_isEmptyStack CppDFSTestVisitor::isEmptyStack;



    CppDFSTestVisitor::_pop CppDFSTestVisitor::pop;



    CppDFSTestVisitor::_push CppDFSTestVisitor::push;



    CppDFSTestVisitor::_pushPopTopBehavior CppDFSTestVisitor::pushPopTopBehavior;



    CppDFSTestVisitor::_top CppDFSTestVisitor::top;

    pair<CppDFSTestVisitor::OutEdgeIterator, CppDFSTestVisitor::OutEdgeIterator> CppDFSTestVisitor::__pair;



    CppDFSTestVisitor::_iterRangeBegin CppDFSTestVisitor::iterRangeBegin;



    CppDFSTestVisitor::_iterRangeEnd CppDFSTestVisitor::iterRangeEnd;



    CppDFSTestVisitor::_makeOutEdgeIteratorRange CppDFSTestVisitor::makeOutEdgeIteratorRange;



    CppDFSTestVisitor::_outEdgeIterEnd CppDFSTestVisitor::outEdgeIterEnd;



    CppDFSTestVisitor::_outEdgeIterNext CppDFSTestVisitor::outEdgeIterNext;



    CppDFSTestVisitor::_depthFirstSearch CppDFSTestVisitor::depthFirstSearch;



    CppDFSTestVisitor::_examineVertex CppDFSTestVisitor::examineVertex;



    CppDFSTestVisitor::_numVertices CppDFSTestVisitor::numVertices;



    CppDFSTestVisitor::_outDegree CppDFSTestVisitor::outDegree;



    CppDFSTestVisitor::_outEdges CppDFSTestVisitor::outEdges;



    CppDFSTestVisitor::_toVertexDescriptor CppDFSTestVisitor::toVertexDescriptor;



    CppDFSTestVisitor::_vertices CppDFSTestVisitor::vertices;



    CppDFSTestVisitor::_edgeIterEnd CppDFSTestVisitor::edgeIterEnd;



    CppDFSTestVisitor::_edgeIterNext CppDFSTestVisitor::edgeIterNext;



    CppDFSTestVisitor::_edges CppDFSTestVisitor::edges;



    CppDFSTestVisitor::_defaultAction CppDFSTestVisitor::defaultAction;



    CppDFSTestVisitor::_edgeIterUnpack CppDFSTestVisitor::edgeIterUnpack;



    CppDFSTestVisitor::_outEdgeIterUnpack CppDFSTestVisitor::outEdgeIterUnpack;



    CppDFSTestVisitor::_src CppDFSTestVisitor::src;



    CppDFSTestVisitor::_tgt CppDFSTestVisitor::tgt;



    CppDFSTestVisitor::_toEdgeDescriptor CppDFSTestVisitor::toEdgeDescriptor;



    CppDFSTestVisitor::_makeEdge CppDFSTestVisitor::makeEdge;



    CppDFSTestVisitor::_bfsInnerLoopRepeat CppDFSTestVisitor::bfsInnerLoopRepeat;



    for_iterator_loop3_2<CppDFSTestVisitor::Graph, CppDFSTestVisitor::VertexDescriptor, CppDFSTestVisitor::OutEdgeIterator, CppDFSTestVisitor::VertexVector, CppDFSTestVisitor::Stack, CppDFSTestVisitor::ColorPropertyMap, CppDFSTestVisitor::_outEdgeIterEnd, CppDFSTestVisitor::_outEdgeIterNext, CppDFSTestVisitor::_bfsInnerLoopStep> CppDFSTestVisitor::__for_iterator_loop3_2;

    CppDFSTestVisitor::_bfsInnerLoopStep CppDFSTestVisitor::bfsInnerLoopStep;



    CppDFSTestVisitor::_bfsOuterLoopCond CppDFSTestVisitor::bfsOuterLoopCond;



    CppDFSTestVisitor::_bfsOuterLoopRepeat CppDFSTestVisitor::bfsOuterLoopRepeat;



    while_loop3<CppDFSTestVisitor::Graph, CppDFSTestVisitor::VertexVector, CppDFSTestVisitor::Stack, CppDFSTestVisitor::ColorPropertyMap, CppDFSTestVisitor::_bfsOuterLoopCond, CppDFSTestVisitor::_bfsOuterLoopStep> CppDFSTestVisitor::__while_loop3;

    CppDFSTestVisitor::_bfsOuterLoopStep CppDFSTestVisitor::bfsOuterLoopStep;



    CppDFSTestVisitor::_breadthFirstVisit CppDFSTestVisitor::breadthFirstVisit;



    CppDFSTestVisitor::_black CppDFSTestVisitor::black;



    CppDFSTestVisitor::_get CppDFSTestVisitor::get;



    CppDFSTestVisitor::_gray CppDFSTestVisitor::gray;



    CppDFSTestVisitor::_initMap CppDFSTestVisitor::initMap;



    CppDFSTestVisitor::_put CppDFSTestVisitor::put;



    CppDFSTestVisitor::_white CppDFSTestVisitor::white;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_types CppDijkstraVisitor::__base_types;

    base_float_ops CppDijkstraVisitor::__base_float_ops;



    CppDijkstraVisitor::_emptyVertexVector CppDijkstraVisitor::emptyVertexVector;

    vector<CppDijkstraVisitor::VertexDescriptor> CppDijkstraVisitor::__vector;



    CppDijkstraVisitor::_pushBack CppDijkstraVisitor::pushBack;



    CppDijkstraVisitor::_vertexIterEnd CppDijkstraVisitor::vertexIterEnd;



    CppDijkstraVisitor::_vertexIterNext CppDijkstraVisitor::vertexIterNext;





    CppDijkstraVisitor::_emptyVPMap CppDijkstraVisitor::emptyVPMap;



    CppDijkstraVisitor::_forIterationEnd CppDijkstraVisitor::forIterationEnd;



    CppDijkstraVisitor::_populateVPMapLoopRepeat CppDijkstraVisitor::populateVPMapLoopRepeat;



    for_iterator_loop<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_populateVPMapLoopStep> CppDijkstraVisitor::__for_iterator_loop;

    CppDijkstraVisitor::_populateVPMapLoopStep CppDijkstraVisitor::populateVPMapLoopStep;

    read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> CppDijkstraVisitor::__read_write_property_map1;

    two_bit_color_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> CppDijkstraVisitor::__two_bit_color_map;

    CppDijkstraVisitor::_vertexIterUnpack CppDijkstraVisitor::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppDijkstraVisitor::Vertex> CppDijkstraVisitor::__incidence_and_vertex_list_and_edge_list_graph;

    pair<CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::OutEdgeIterator> CppDijkstraVisitor::__pair;



    CppDijkstraVisitor::_iterRangeBegin CppDijkstraVisitor::iterRangeBegin;



    CppDijkstraVisitor::_iterRangeEnd CppDijkstraVisitor::iterRangeEnd;



    CppDijkstraVisitor::_makeOutEdgeIteratorRange CppDijkstraVisitor::makeOutEdgeIteratorRange;



    CppDijkstraVisitor::_outEdgeIterEnd CppDijkstraVisitor::outEdgeIterEnd;



    CppDijkstraVisitor::_outEdgeIterNext CppDijkstraVisitor::outEdgeIterNext;



    CppDijkstraVisitor::_numVertices CppDijkstraVisitor::numVertices;



    CppDijkstraVisitor::_outDegree CppDijkstraVisitor::outDegree;



    CppDijkstraVisitor::_outEdges CppDijkstraVisitor::outEdges;



    CppDijkstraVisitor::_toVertexDescriptor CppDijkstraVisitor::toVertexDescriptor;



    CppDijkstraVisitor::_vertices CppDijkstraVisitor::vertices;



    CppDijkstraVisitor::_edgeIterEnd CppDijkstraVisitor::edgeIterEnd;



    CppDijkstraVisitor::_edgeIterNext CppDijkstraVisitor::edgeIterNext;



    CppDijkstraVisitor::_edges CppDijkstraVisitor::edges;



    CppDijkstraVisitor::_edgeIterUnpack CppDijkstraVisitor::edgeIterUnpack;



    CppDijkstraVisitor::_outEdgeIterUnpack CppDijkstraVisitor::outEdgeIterUnpack;



    CppDijkstraVisitor::_src CppDijkstraVisitor::src;



    CppDijkstraVisitor::_tgt CppDijkstraVisitor::tgt;



    CppDijkstraVisitor::_toEdgeDescriptor CppDijkstraVisitor::toEdgeDescriptor;



    CppDijkstraVisitor::_makeEdge CppDijkstraVisitor::makeEdge;



    CppDijkstraVisitor::_emptyECMap CppDijkstraVisitor::emptyECMap;



    CppDijkstraVisitor::_getEdgeCostMap CppDijkstraVisitor::getEdgeCostMap;



    CppDijkstraVisitor::_getVertexPredecessorMap CppDijkstraVisitor::getVertexPredecessorMap;



    CppDijkstraVisitor::_putVertexPredecessorMap CppDijkstraVisitor::putVertexPredecessorMap;

    triplet<CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::VertexPredecessorMap, CppDijkstraVisitor::EdgeCostMap> CppDijkstraVisitor::__triplet;



    CppDijkstraVisitor::_emptyVCMap CppDijkstraVisitor::emptyVCMap;



    CppDijkstraVisitor::_getVertexCostMap CppDijkstraVisitor::getVertexCostMap;



    CppDijkstraVisitor::_makeStateWithMaps CppDijkstraVisitor::makeStateWithMaps;



    CppDijkstraVisitor::_putVertexCostMap CppDijkstraVisitor::putVertexCostMap;



    CppDijkstraVisitor::_relax CppDijkstraVisitor::relax;

    read_write_property_map<CppDijkstraVisitor::EdgeDescriptor, CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_outEdgeIterEnd, CppDijkstraVisitor::_outEdgeIterNext, CppDijkstraVisitor::_outEdgeIterUnpack> CppDijkstraVisitor::__read_write_property_map;

    read_write_property_map<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::VertexIterator, CppDijkstraVisitor::Cost, CppDijkstraVisitor::_vertexIterEnd, CppDijkstraVisitor::_vertexIterNext, CppDijkstraVisitor::_vertexIterUnpack> CppDijkstraVisitor::__read_write_property_map0;



    CppDijkstraVisitor::_dijkstraShortestPaths CppDijkstraVisitor::dijkstraShortestPaths;



    CppDijkstraVisitor::_inf CppDijkstraVisitor::inf;



    CppDijkstraVisitor::_less CppDijkstraVisitor::less;



    CppDijkstraVisitor::_negate CppDijkstraVisitor::negate;



    CppDijkstraVisitor::_plus CppDijkstraVisitor::plus;



    CppDijkstraVisitor::_zero CppDijkstraVisitor::zero;



    CppDijkstraVisitor::_black CppDijkstraVisitor::black;





    CppDijkstraVisitor::_bfsInnerLoopRepeat CppDijkstraVisitor::bfsInnerLoopRepeat;



    for_iterator_loop3_2<CppDijkstraVisitor::Graph, CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::OutEdgeIterator, CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap, CppDijkstraVisitor::_outEdgeIterEnd, CppDijkstraVisitor::_outEdgeIterNext, CppDijkstraVisitor::_bfsInnerLoopStep> CppDijkstraVisitor::__for_iterator_loop3_2;

    CppDijkstraVisitor::_bfsInnerLoopStep CppDijkstraVisitor::bfsInnerLoopStep;



    CppDijkstraVisitor::_bfsOuterLoopCond CppDijkstraVisitor::bfsOuterLoopCond;



    CppDijkstraVisitor::_bfsOuterLoopRepeat CppDijkstraVisitor::bfsOuterLoopRepeat;



    while_loop3<CppDijkstraVisitor::Graph, CppDijkstraVisitor::StateWithMaps, CppDijkstraVisitor::PriorityQueue, CppDijkstraVisitor::ColorPropertyMap, CppDijkstraVisitor::_bfsOuterLoopCond, CppDijkstraVisitor::_bfsOuterLoopStep> CppDijkstraVisitor::__while_loop3;

    CppDijkstraVisitor::_bfsOuterLoopStep CppDijkstraVisitor::bfsOuterLoopStep;



    CppDijkstraVisitor::_blackTarget CppDijkstraVisitor::blackTarget;



    CppDijkstraVisitor::_breadthFirstVisit CppDijkstraVisitor::breadthFirstVisit;



    CppDijkstraVisitor::_discoverVertex CppDijkstraVisitor::discoverVertex;



    CppDijkstraVisitor::_emptyPriorityQueue CppDijkstraVisitor::emptyPriorityQueue;



    CppDijkstraVisitor::_examineEdge CppDijkstraVisitor::examineEdge;



    CppDijkstraVisitor::_examineVertex CppDijkstraVisitor::examineVertex;



    CppDijkstraVisitor::_finishVertex CppDijkstraVisitor::finishVertex;



    CppDijkstraVisitor::_front CppDijkstraVisitor::front;



    CppDijkstraVisitor::_grayTarget CppDijkstraVisitor::grayTarget;



    CppDijkstraVisitor::_isEmptyQueue CppDijkstraVisitor::isEmptyQueue;



    CppDijkstraVisitor::_nonTreeEdge CppDijkstraVisitor::nonTreeEdge;



    CppDijkstraVisitor::_pop CppDijkstraVisitor::pop;



    CppDijkstraVisitor::_push CppDijkstraVisitor::push;



    CppDijkstraVisitor::_treeEdge CppDijkstraVisitor::treeEdge;



    CppDijkstraVisitor::_update CppDijkstraVisitor::update;

    priority_queue<CppDijkstraVisitor::VertexDescriptor, CppDijkstraVisitor::Cost, CppDijkstraVisitor::VertexCostMap, CppDijkstraVisitor::_get> CppDijkstraVisitor::__priority_queue;

    CppDijkstraVisitor::_get CppDijkstraVisitor::get;



    CppDijkstraVisitor::_gray CppDijkstraVisitor::gray;



    CppDijkstraVisitor::_initMap CppDijkstraVisitor::initMap;



    CppDijkstraVisitor::_put CppDijkstraVisitor::put;



    CppDijkstraVisitor::_white CppDijkstraVisitor::white;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_unit CppJohnson::__base_unit;

    base_types CppJohnson::__base_types;

    base_float_ops CppJohnson::__base_float_ops;

    base_bool CppJohnson::__base_bool;



    CppJohnson::_emptyVertexVector CppJohnson::emptyVertexVector;

    vector<CppJohnson::VertexDescriptor> CppJohnson::__vector;



    CppJohnson::_pushBack CppJohnson::pushBack;



    CppJohnson::_vertexIterEnd CppJohnson::vertexIterEnd;



    CppJohnson::_vertexIterNext CppJohnson::vertexIterNext;





    CppJohnson::_emptyVPMap CppJohnson::emptyVPMap;

    read_write_property_map<CppJohnson::VertexDescriptor, CppJohnson::VertexIterator, CppJohnson::VertexDescriptor, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_vertexIterUnpack> CppJohnson::__read_write_property_map2;

    two_bit_color_map<CppJohnson::VertexDescriptor, CppJohnson::VertexIterator, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_vertexIterUnpack> CppJohnson::__two_bit_color_map;

    CppJohnson::_vertexIterUnpack CppJohnson::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppJohnson::Vertex> CppJohnson::__incidence_and_vertex_list_and_edge_list_graph;



    CppJohnson::_forIterationEnd CppJohnson::forIterationEnd;



    CppJohnson::_populateVPMapLoopRepeat CppJohnson::populateVPMapLoopRepeat;



    for_iterator_loop<CppJohnson::Unit, CppJohnson::VertexIterator, CppJohnson::VertexPredecessorMap, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_populateVPMapLoopStep> CppJohnson::__for_iterator_loop0;

    for_iterator_loop<CppJohnson::VertexDescriptor, CppJohnson::VertexIterator, CppJohnson::VertexPredecessorMap, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_populateVPMapLoopStep> CppJohnson::__for_iterator_loop1;

    CppJohnson::_populateVPMapLoopStep CppJohnson::populateVPMapLoopStep;



    CppJohnson::_unit CppJohnson::unit;

    pair<CppJohnson::OutEdgeIterator, CppJohnson::OutEdgeIterator> CppJohnson::__pair;



    CppJohnson::_iterRangeBegin CppJohnson::iterRangeBegin;



    CppJohnson::_iterRangeEnd CppJohnson::iterRangeEnd;



    CppJohnson::_makeOutEdgeIteratorRange CppJohnson::makeOutEdgeIteratorRange;



    CppJohnson::_outEdgeIterEnd CppJohnson::outEdgeIterEnd;



    CppJohnson::_outEdgeIterNext CppJohnson::outEdgeIterNext;



    CppJohnson::_numVertices CppJohnson::numVertices;



    CppJohnson::_outDegree CppJohnson::outDegree;



    CppJohnson::_outEdges CppJohnson::outEdges;



    CppJohnson::_toVertexDescriptor CppJohnson::toVertexDescriptor;



    CppJohnson::_vertices CppJohnson::vertices;



    CppJohnson::_edgeIterEnd CppJohnson::edgeIterEnd;



    CppJohnson::_edgeIterNext CppJohnson::edgeIterNext;



    CppJohnson::_edges CppJohnson::edges;



    CppJohnson::_edgeIterUnpack CppJohnson::edgeIterUnpack;



    CppJohnson::_edgeMinimized CppJohnson::edgeMinimized;



    CppJohnson::_edgeNotMinimized CppJohnson::edgeNotMinimized;



    CppJohnson::_edgeNotRelaxed CppJohnson::edgeNotRelaxed;



    CppJohnson::_edgeRelaxed CppJohnson::edgeRelaxed;



    CppJohnson::_outEdgeIterUnpack CppJohnson::outEdgeIterUnpack;



    CppJohnson::_src CppJohnson::src;



    CppJohnson::_tgt CppJohnson::tgt;



    CppJohnson::_toEdgeDescriptor CppJohnson::toEdgeDescriptor;



    CppJohnson::_makeEdge CppJohnson::makeEdge;



    CppJohnson::_emptyECMap CppJohnson::emptyECMap;



    CppJohnson::_getEdgeCostMap CppJohnson::getEdgeCostMap;



    CppJohnson::_getVertexPredecessorMap CppJohnson::getVertexPredecessorMap;



    CppJohnson::_putVertexPredecessorMap CppJohnson::putVertexPredecessorMap;



    CppJohnson::_emptyVCMatrix CppJohnson::emptyVCMatrix;

    read_write_property_map<CppJohnson::VertexDescriptor, CppJohnson::VertexIterator, CppJohnson::VertexCostMap, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_vertexIterUnpack> CppJohnson::__read_write_property_map1;

    triplet<CppJohnson::VertexCostMap, CppJohnson::VertexPredecessorMap, CppJohnson::EdgeCostMap> CppJohnson::__triplet;



    CppJohnson::_adjustVertexLoopRepeat CppJohnson::adjustVertexLoopRepeat;



    for_iterator_loop1_2<CppJohnson::VertexDescriptor, CppJohnson::VertexCostMap, CppJohnson::VertexIterator, CppJohnson::VertexCostMap, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_adjustVertexLoopStep> CppJohnson::__for_iterator_loop1_2;

    CppJohnson::_adjustVertexLoopStep CppJohnson::adjustVertexLoopStep;



    CppJohnson::_dijkstraAndAdjustLoopRepeat CppJohnson::dijkstraAndAdjustLoopRepeat;



    for_iterator_loop1_3<CppJohnson::EdgeCostMap, CppJohnson::VertexCostMap, CppJohnson::Graph, CppJohnson::VertexIterator, CppJohnson::VertexCostMatrix, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_dijkstraAndAdjustLoopStep> CppJohnson::__for_iterator_loop1_30;

    CppJohnson::_dijkstraAndAdjustLoopStep CppJohnson::dijkstraAndAdjustLoopStep;



    CppJohnson::_edgeRelaxationInnerLoopRepeat CppJohnson::edgeRelaxationInnerLoopRepeat;



    for_iterator_loop3_2<CppJohnson::EdgeCostMap, CppJohnson::Graph, CppJohnson::EdgeIterator, CppJohnson::Unit, CppJohnson::VertexCostMap, CppJohnson::VertexPredecessorMap, CppJohnson::_edgeIterEnd, CppJohnson::_edgeIterNext, CppJohnson::_edgeRelaxationInnerLoopStep> CppJohnson::__for_iterator_loop3_2;

    CppJohnson::_edgeRelaxationInnerLoopStep CppJohnson::edgeRelaxationInnerLoopStep;



    CppJohnson::_edgeRelaxationOuterLoopRepeat CppJohnson::edgeRelaxationOuterLoopRepeat;



    for_iterator_loop3_2<CppJohnson::EdgeCostMap, CppJohnson::Graph, CppJohnson::VertexIterator, CppJohnson::Unit, CppJohnson::VertexCostMap, CppJohnson::VertexPredecessorMap, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_edgeRelaxationOuterLoopStep> CppJohnson::__for_iterator_loop3_20;

    CppJohnson::_edgeRelaxationOuterLoopStep CppJohnson::edgeRelaxationOuterLoopStep;



    CppJohnson::_emptyVCMap CppJohnson::emptyVCMap;



    CppJohnson::_getVertexCostMap CppJohnson::getVertexCostMap;



    CppJohnson::_makeStateWithMaps CppJohnson::makeStateWithMaps;



    CppJohnson::_putVertexCostMap CppJohnson::putVertexCostMap;



    CppJohnson::_relax CppJohnson::relax;



    CppJohnson::_reweightEdgeLoopRepeat CppJohnson::reweightEdgeLoopRepeat;



    for_iterator_loop1_3<CppJohnson::EdgeCostMap, CppJohnson::VertexCostMap, CppJohnson::Graph, CppJohnson::EdgeIterator, CppJohnson::EdgeCostMap, CppJohnson::_edgeIterEnd, CppJohnson::_edgeIterNext, CppJohnson::_reweightEdgeLoopStep> CppJohnson::__for_iterator_loop1_3;

    CppJohnson::_reweightEdgeLoopStep CppJohnson::reweightEdgeLoopStep;

    read_write_property_map<CppJohnson::EdgeDescriptor, CppJohnson::OutEdgeIterator, CppJohnson::Cost, CppJohnson::_outEdgeIterEnd, CppJohnson::_outEdgeIterNext, CppJohnson::_outEdgeIterUnpack> CppJohnson::__read_write_property_map;

    read_write_property_map<CppJohnson::VertexDescriptor, CppJohnson::VertexIterator, CppJohnson::Cost, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_vertexIterUnpack> CppJohnson::__read_write_property_map0;



    CppJohnson::_dijkstraShortestPaths CppJohnson::dijkstraShortestPaths;



    CppJohnson::_inf CppJohnson::inf;



    CppJohnson::_initializeVertexCostMapLoopRepeat CppJohnson::initializeVertexCostMapLoopRepeat;



    for_iterator_loop<CppJohnson::Cost, CppJohnson::VertexIterator, CppJohnson::VertexCostMap, CppJohnson::_vertexIterEnd, CppJohnson::_vertexIterNext, CppJohnson::_initializeVertexCostMapLoopStep> CppJohnson::__for_iterator_loop;

    CppJohnson::_initializeVertexCostMapLoopStep CppJohnson::initializeVertexCostMapLoopStep;



    CppJohnson::_less CppJohnson::less;



    CppJohnson::_negate CppJohnson::negate;



    CppJohnson::_plus CppJohnson::plus;



    CppJohnson::_zero CppJohnson::zero;



    CppJohnson::_black CppJohnson::black;





    CppJohnson::_bfsInnerLoopRepeat CppJohnson::bfsInnerLoopRepeat;



    for_iterator_loop3_2<CppJohnson::Graph, CppJohnson::VertexDescriptor, CppJohnson::OutEdgeIterator, CppJohnson::StateWithMaps, CppJohnson::PriorityQueue, CppJohnson::ColorPropertyMap, CppJohnson::_outEdgeIterEnd, CppJohnson::_outEdgeIterNext, CppJohnson::_bfsInnerLoopStep> CppJohnson::__for_iterator_loop3_21;

    CppJohnson::_bfsInnerLoopStep CppJohnson::bfsInnerLoopStep;



    CppJohnson::_bfsOuterLoopCond CppJohnson::bfsOuterLoopCond;



    CppJohnson::_bfsOuterLoopRepeat CppJohnson::bfsOuterLoopRepeat;



    while_loop3<CppJohnson::Graph, CppJohnson::StateWithMaps, CppJohnson::PriorityQueue, CppJohnson::ColorPropertyMap, CppJohnson::_bfsOuterLoopCond, CppJohnson::_bfsOuterLoopStep> CppJohnson::__while_loop3;

    CppJohnson::_bfsOuterLoopStep CppJohnson::bfsOuterLoopStep;



    CppJohnson::_blackTarget CppJohnson::blackTarget;



    CppJohnson::_breadthFirstVisit CppJohnson::breadthFirstVisit;



    CppJohnson::_discoverVertex CppJohnson::discoverVertex;



    CppJohnson::_emptyPriorityQueue CppJohnson::emptyPriorityQueue;



    CppJohnson::_examineEdge CppJohnson::examineEdge;



    CppJohnson::_examineVertex CppJohnson::examineVertex;



    CppJohnson::_finishVertex CppJohnson::finishVertex;



    CppJohnson::_front CppJohnson::front;



    CppJohnson::_grayTarget CppJohnson::grayTarget;



    CppJohnson::_isEmptyQueue CppJohnson::isEmptyQueue;



    CppJohnson::_nonTreeEdge CppJohnson::nonTreeEdge;



    CppJohnson::_pop CppJohnson::pop;



    CppJohnson::_push CppJohnson::push;



    CppJohnson::_treeEdge CppJohnson::treeEdge;



    CppJohnson::_update CppJohnson::update;

    priority_queue<CppJohnson::VertexDescriptor, CppJohnson::Cost, CppJohnson::VertexCostMap, CppJohnson::_get> CppJohnson::__priority_queue;

    CppJohnson::_get CppJohnson::get;



    CppJohnson::_gray CppJohnson::gray;



    CppJohnson::_initMap CppJohnson::initMap;



    CppJohnson::_put CppJohnson::put;



    CppJohnson::_white CppJohnson::white;



    CppJohnson::_bellmanFordShortestPaths CppJohnson::bellmanFordShortestPaths;



    CppJohnson::_bfalse CppJohnson::bfalse;



    CppJohnson::_btrue CppJohnson::btrue;



    CppJohnson::_checkNegativeCycleLoopRepeat CppJohnson::checkNegativeCycleLoopRepeat;



    for_iterator_loop2_3<CppJohnson::VertexCostMap, CppJohnson::EdgeCostMap, CppJohnson::Graph, CppJohnson::EdgeIterator, CppJohnson::Unit, CppJohnson::Bool, CppJohnson::_edgeIterEnd, CppJohnson::_edgeIterNext, CppJohnson::_checkNegativeCycleLoopStep> CppJohnson::__for_iterator_loop2_3;

    CppJohnson::_checkNegativeCycleLoopStep CppJohnson::checkNegativeCycleLoopStep;



    CppJohnson::_holds CppJohnson::holds;



    CppJohnson::_johnsonAllPairsShortestPaths CppJohnson::johnsonAllPairsShortestPaths;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_types CppParallelBFSTestVisitor::__base_types;



    CppParallelBFSTestVisitor::_emptyVertexVector CppParallelBFSTestVisitor::emptyVertexVector;

    thread_safe_fifo_queue<CppParallelBFSTestVisitor::VertexDescriptor> CppParallelBFSTestVisitor::__thread_safe_fifo_queue;

    thread_safe_vector<CppParallelBFSTestVisitor::VertexDescriptor> CppParallelBFSTestVisitor::__thread_safe_vector;



    CppParallelBFSTestVisitor::_pushBack CppParallelBFSTestVisitor::pushBack;



    CppParallelBFSTestVisitor::_vertexIterEnd CppParallelBFSTestVisitor::vertexIterEnd;



    CppParallelBFSTestVisitor::_vertexIterNext CppParallelBFSTestVisitor::vertexIterNext;



    two_bit_color_map<CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::VertexIterator, CppParallelBFSTestVisitor::_vertexIterEnd, CppParallelBFSTestVisitor::_vertexIterNext, CppParallelBFSTestVisitor::_vertexIterUnpack> CppParallelBFSTestVisitor::__two_bit_color_map;

    CppParallelBFSTestVisitor::_vertexIterUnpack CppParallelBFSTestVisitor::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppParallelBFSTestVisitor::Vertex> CppParallelBFSTestVisitor::__incidence_and_vertex_list_and_edge_list_graph;

    pair<CppParallelBFSTestVisitor::OutEdgeIterator, CppParallelBFSTestVisitor::OutEdgeIterator> CppParallelBFSTestVisitor::__pair;



    CppParallelBFSTestVisitor::_iterRangeBegin CppParallelBFSTestVisitor::iterRangeBegin;



    CppParallelBFSTestVisitor::_iterRangeEnd CppParallelBFSTestVisitor::iterRangeEnd;



    CppParallelBFSTestVisitor::_makeOutEdgeIteratorRange CppParallelBFSTestVisitor::makeOutEdgeIteratorRange;



    CppParallelBFSTestVisitor::_outEdgeIterEnd CppParallelBFSTestVisitor::outEdgeIterEnd;



    CppParallelBFSTestVisitor::_outEdgeIterNext CppParallelBFSTestVisitor::outEdgeIterNext;



    CppParallelBFSTestVisitor::_breadthFirstSearch CppParallelBFSTestVisitor::breadthFirstSearch;



    CppParallelBFSTestVisitor::_numVertices CppParallelBFSTestVisitor::numVertices;



    CppParallelBFSTestVisitor::_outDegree CppParallelBFSTestVisitor::outDegree;



    CppParallelBFSTestVisitor::_outEdges CppParallelBFSTestVisitor::outEdges;



    CppParallelBFSTestVisitor::_toVertexDescriptor CppParallelBFSTestVisitor::toVertexDescriptor;



    CppParallelBFSTestVisitor::_vertices CppParallelBFSTestVisitor::vertices;



    CppParallelBFSTestVisitor::_empty CppParallelBFSTestVisitor::empty;



    CppParallelBFSTestVisitor::_examineVertex CppParallelBFSTestVisitor::examineVertex;



    CppParallelBFSTestVisitor::_front CppParallelBFSTestVisitor::front;



    CppParallelBFSTestVisitor::_isEmptyQueue CppParallelBFSTestVisitor::isEmptyQueue;



    CppParallelBFSTestVisitor::_pop CppParallelBFSTestVisitor::pop;



    CppParallelBFSTestVisitor::_push CppParallelBFSTestVisitor::push;



    CppParallelBFSTestVisitor::_pushPopBehavior CppParallelBFSTestVisitor::pushPopBehavior;



    CppParallelBFSTestVisitor::_edgeIterEnd CppParallelBFSTestVisitor::edgeIterEnd;



    CppParallelBFSTestVisitor::_edgeIterNext CppParallelBFSTestVisitor::edgeIterNext;



    CppParallelBFSTestVisitor::_edges CppParallelBFSTestVisitor::edges;



    CppParallelBFSTestVisitor::_defaultAction CppParallelBFSTestVisitor::defaultAction;



    CppParallelBFSTestVisitor::_edgeIterUnpack CppParallelBFSTestVisitor::edgeIterUnpack;



    CppParallelBFSTestVisitor::_outEdgeIterUnpack CppParallelBFSTestVisitor::outEdgeIterUnpack;



    CppParallelBFSTestVisitor::_src CppParallelBFSTestVisitor::src;



    CppParallelBFSTestVisitor::_tgt CppParallelBFSTestVisitor::tgt;



    CppParallelBFSTestVisitor::_toEdgeDescriptor CppParallelBFSTestVisitor::toEdgeDescriptor;



    CppParallelBFSTestVisitor::_makeEdge CppParallelBFSTestVisitor::makeEdge;



    CppParallelBFSTestVisitor::_bfsInnerLoopRepeat CppParallelBFSTestVisitor::bfsInnerLoopRepeat;



    for_parallel_iterator_loop3_2<CppParallelBFSTestVisitor::Graph, CppParallelBFSTestVisitor::VertexDescriptor, CppParallelBFSTestVisitor::OutEdgeIterator, CppParallelBFSTestVisitor::VertexVector, CppParallelBFSTestVisitor::FIFOQueue, CppParallelBFSTestVisitor::ColorPropertyMap, CppParallelBFSTestVisitor::_outEdgeIterEnd, CppParallelBFSTestVisitor::_outEdgeIterNext, CppParallelBFSTestVisitor::_bfsInnerLoopStep> CppParallelBFSTestVisitor::__for_parallel_iterator_loop3_2;

    CppParallelBFSTestVisitor::_bfsInnerLoopStep CppParallelBFSTestVisitor::bfsInnerLoopStep;



    CppParallelBFSTestVisitor::_bfsOuterLoopCond CppParallelBFSTestVisitor::bfsOuterLoopCond;



    CppParallelBFSTestVisitor::_bfsOuterLoopRepeat CppParallelBFSTestVisitor::bfsOuterLoopRepeat;



    while_loop3<CppParallelBFSTestVisitor::Graph, CppParallelBFSTestVisitor::VertexVector, CppParallelBFSTestVisitor::FIFOQueue, CppParallelBFSTestVisitor::ColorPropertyMap, CppParallelBFSTestVisitor::_bfsOuterLoopCond, CppParallelBFSTestVisitor::_bfsOuterLoopStep> CppParallelBFSTestVisitor::__while_loop3;

    CppParallelBFSTestVisitor::_bfsOuterLoopStep CppParallelBFSTestVisitor::bfsOuterLoopStep;



    CppParallelBFSTestVisitor::_breadthFirstVisit CppParallelBFSTestVisitor::breadthFirstVisit;



    CppParallelBFSTestVisitor::_black CppParallelBFSTestVisitor::black;



    CppParallelBFSTestVisitor::_get CppParallelBFSTestVisitor::get;



    CppParallelBFSTestVisitor::_gray CppParallelBFSTestVisitor::gray;



    CppParallelBFSTestVisitor::_initMap CppParallelBFSTestVisitor::initMap;



    CppParallelBFSTestVisitor::_put CppParallelBFSTestVisitor::put;



    CppParallelBFSTestVisitor::_white CppParallelBFSTestVisitor::white;

} // examples
} // bgl
} // mg_src
} // bgl_cpp

namespace examples {
namespace bgl {
namespace mg_src {
namespace bgl_cpp {
    base_types CppPrimVisitor::__base_types;

    base_float_ops CppPrimVisitor::__base_float_ops;



    CppPrimVisitor::_emptyVertexVector CppPrimVisitor::emptyVertexVector;

    vector<CppPrimVisitor::VertexDescriptor> CppPrimVisitor::__vector;



    CppPrimVisitor::_pushBack CppPrimVisitor::pushBack;



    CppPrimVisitor::_vertexIterEnd CppPrimVisitor::vertexIterEnd;



    CppPrimVisitor::_vertexIterNext CppPrimVisitor::vertexIterNext;





    CppPrimVisitor::_emptyVPMap CppPrimVisitor::emptyVPMap;



    CppPrimVisitor::_forIterationEnd CppPrimVisitor::forIterationEnd;



    CppPrimVisitor::_populateVPMapLoopRepeat CppPrimVisitor::populateVPMapLoopRepeat;



    for_iterator_loop<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::VertexPredecessorMap, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_populateVPMapLoopStep> CppPrimVisitor::__for_iterator_loop;

    CppPrimVisitor::_populateVPMapLoopStep CppPrimVisitor::populateVPMapLoopStep;

    read_write_property_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::VertexDescriptor, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack> CppPrimVisitor::__read_write_property_map1;

    two_bit_color_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack> CppPrimVisitor::__two_bit_color_map;

    CppPrimVisitor::_vertexIterUnpack CppPrimVisitor::vertexIterUnpack;

    incidence_and_vertex_list_and_edge_list_graph<CppPrimVisitor::Vertex> CppPrimVisitor::__incidence_and_vertex_list_and_edge_list_graph;

    pair<CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::OutEdgeIterator> CppPrimVisitor::__pair;



    CppPrimVisitor::_iterRangeBegin CppPrimVisitor::iterRangeBegin;



    CppPrimVisitor::_iterRangeEnd CppPrimVisitor::iterRangeEnd;



    CppPrimVisitor::_makeOutEdgeIteratorRange CppPrimVisitor::makeOutEdgeIteratorRange;



    CppPrimVisitor::_outEdgeIterEnd CppPrimVisitor::outEdgeIterEnd;



    CppPrimVisitor::_outEdgeIterNext CppPrimVisitor::outEdgeIterNext;



    CppPrimVisitor::_numVertices CppPrimVisitor::numVertices;



    CppPrimVisitor::_outDegree CppPrimVisitor::outDegree;



    CppPrimVisitor::_outEdges CppPrimVisitor::outEdges;



    CppPrimVisitor::_toVertexDescriptor CppPrimVisitor::toVertexDescriptor;



    CppPrimVisitor::_vertices CppPrimVisitor::vertices;



    CppPrimVisitor::_edgeIterEnd CppPrimVisitor::edgeIterEnd;



    CppPrimVisitor::_edgeIterNext CppPrimVisitor::edgeIterNext;



    CppPrimVisitor::_edges CppPrimVisitor::edges;



    CppPrimVisitor::_edgeIterUnpack CppPrimVisitor::edgeIterUnpack;



    CppPrimVisitor::_outEdgeIterUnpack CppPrimVisitor::outEdgeIterUnpack;



    CppPrimVisitor::_src CppPrimVisitor::src;



    CppPrimVisitor::_tgt CppPrimVisitor::tgt;



    CppPrimVisitor::_toEdgeDescriptor CppPrimVisitor::toEdgeDescriptor;



    CppPrimVisitor::_makeEdge CppPrimVisitor::makeEdge;



    CppPrimVisitor::_emptyECMap CppPrimVisitor::emptyECMap;



    CppPrimVisitor::_getEdgeCostMap CppPrimVisitor::getEdgeCostMap;



    CppPrimVisitor::_getVertexPredecessorMap CppPrimVisitor::getVertexPredecessorMap;



    CppPrimVisitor::_putVertexPredecessorMap CppPrimVisitor::putVertexPredecessorMap;

    triplet<CppPrimVisitor::VertexCostMap, CppPrimVisitor::VertexPredecessorMap, CppPrimVisitor::EdgeCostMap> CppPrimVisitor::__triplet;



    CppPrimVisitor::_emptyVCMap CppPrimVisitor::emptyVCMap;



    CppPrimVisitor::_getVertexCostMap CppPrimVisitor::getVertexCostMap;



    CppPrimVisitor::_makeStateWithMaps CppPrimVisitor::makeStateWithMaps;



    CppPrimVisitor::_putVertexCostMap CppPrimVisitor::putVertexCostMap;



    CppPrimVisitor::_relax CppPrimVisitor::relax;

    read_write_property_map<CppPrimVisitor::EdgeDescriptor, CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::Cost, CppPrimVisitor::_outEdgeIterEnd, CppPrimVisitor::_outEdgeIterNext, CppPrimVisitor::_outEdgeIterUnpack> CppPrimVisitor::__read_write_property_map;

    read_write_property_map<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::VertexIterator, CppPrimVisitor::Cost, CppPrimVisitor::_vertexIterEnd, CppPrimVisitor::_vertexIterNext, CppPrimVisitor::_vertexIterUnpack> CppPrimVisitor::__read_write_property_map0;



    CppPrimVisitor::_inf CppPrimVisitor::inf;



    CppPrimVisitor::_less CppPrimVisitor::less;



    CppPrimVisitor::_negate CppPrimVisitor::negate;



    CppPrimVisitor::_plus CppPrimVisitor::plus;



    CppPrimVisitor::_primMinimumSpanningTree CppPrimVisitor::primMinimumSpanningTree;



    CppPrimVisitor::_second CppPrimVisitor::second;



    CppPrimVisitor::_zero CppPrimVisitor::zero;



    CppPrimVisitor::_black CppPrimVisitor::black;





    CppPrimVisitor::_bfsInnerLoopRepeat CppPrimVisitor::bfsInnerLoopRepeat;



    for_iterator_loop3_2<CppPrimVisitor::Graph, CppPrimVisitor::VertexDescriptor, CppPrimVisitor::OutEdgeIterator, CppPrimVisitor::StateWithMaps, CppPrimVisitor::PriorityQueue, CppPrimVisitor::ColorPropertyMap, CppPrimVisitor::_outEdgeIterEnd, CppPrimVisitor::_outEdgeIterNext, CppPrimVisitor::_bfsInnerLoopStep> CppPrimVisitor::__for_iterator_loop3_2;

    CppPrimVisitor::_bfsInnerLoopStep CppPrimVisitor::bfsInnerLoopStep;



    CppPrimVisitor::_bfsOuterLoopCond CppPrimVisitor::bfsOuterLoopCond;



    CppPrimVisitor::_bfsOuterLoopRepeat CppPrimVisitor::bfsOuterLoopRepeat;



    while_loop3<CppPrimVisitor::Graph, CppPrimVisitor::StateWithMaps, CppPrimVisitor::PriorityQueue, CppPrimVisitor::ColorPropertyMap, CppPrimVisitor::_bfsOuterLoopCond, CppPrimVisitor::_bfsOuterLoopStep> CppPrimVisitor::__while_loop3;

    CppPrimVisitor::_bfsOuterLoopStep CppPrimVisitor::bfsOuterLoopStep;



    CppPrimVisitor::_blackTarget CppPrimVisitor::blackTarget;



    CppPrimVisitor::_breadthFirstVisit CppPrimVisitor::breadthFirstVisit;



    CppPrimVisitor::_discoverVertex CppPrimVisitor::discoverVertex;



    CppPrimVisitor::_emptyPriorityQueue CppPrimVisitor::emptyPriorityQueue;



    CppPrimVisitor::_examineEdge CppPrimVisitor::examineEdge;



    CppPrimVisitor::_examineVertex CppPrimVisitor::examineVertex;



    CppPrimVisitor::_finishVertex CppPrimVisitor::finishVertex;



    CppPrimVisitor::_front CppPrimVisitor::front;



    CppPrimVisitor::_grayTarget CppPrimVisitor::grayTarget;



    CppPrimVisitor::_isEmptyQueue CppPrimVisitor::isEmptyQueue;



    CppPrimVisitor::_nonTreeEdge CppPrimVisitor::nonTreeEdge;



    CppPrimVisitor::_pop CppPrimVisitor::pop;



    CppPrimVisitor::_push CppPrimVisitor::push;



    CppPrimVisitor::_treeEdge CppPrimVisitor::treeEdge;



    CppPrimVisitor::_update CppPrimVisitor::update;

    priority_queue<CppPrimVisitor::VertexDescriptor, CppPrimVisitor::Cost, CppPrimVisitor::VertexCostMap, CppPrimVisitor::_get> CppPrimVisitor::__priority_queue;

    CppPrimVisitor::_get CppPrimVisitor::get;



    CppPrimVisitor::_gray CppPrimVisitor::gray;



    CppPrimVisitor::_initMap CppPrimVisitor::initMap;



    CppPrimVisitor::_put CppPrimVisitor::put;



    CppPrimVisitor::_white CppPrimVisitor::white;

} // examples
} // bgl
} // mg_src
} // bgl_cpp