#pragma once

#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl_v2/mg-src/bgl_v2.hpp"

namespace dijkstra {
// Dijkstra test
using examples::bgl_v2::mg_src::bgl_v2::CppDijkstraVisitor;
typedef CppDijkstraVisitor::Cost Cost;
typedef CppDijkstraVisitor::Edge Edge;
typedef CppDijkstraVisitor::Graph Graph;
typedef CppDijkstraVisitor::Vertex Vertex;
typedef CppDijkstraVisitor::VertexCostMap VertexCostMap;
typedef CppDijkstraVisitor::EdgeCostMap EdgeCostMap;
typedef CppDijkstraVisitor::VertexPredecessorMap VertexPredecessorMap;

auto makeEdge = CppDijkstraVisitor::makeEdge;
auto emptyVertexList = CppDijkstraVisitor::emptyVertexList;
auto head = CppDijkstraVisitor::head;
auto tail = CppDijkstraVisitor::tail;
auto get = CppDijkstraVisitor::get;
auto put = CppDijkstraVisitor::put;


inline std::pair<std::list<CppDijkstraVisitor::Edge>, EdgeCostMap>
        testEdgesWithCost() {
    std::list<CppDijkstraVisitor::Edge> edges;
    EdgeCostMap edgeCostMap = CppDijkstraVisitor::emptyECMap();

    edges.push_back(makeEdge(0, 1));
    edgeCostMap = put(edgeCostMap, makeEdge(0, 1), 0.5);

    edges.push_back(makeEdge(1, 2));
    edgeCostMap = put(edgeCostMap, makeEdge(1, 2), 5.6);

    edges.push_back(makeEdge(1, 3));
    edgeCostMap = put(edgeCostMap, makeEdge(1, 3), 0.2);
    
    edges.push_back(makeEdge(3, 4));
    edgeCostMap = put(edgeCostMap, makeEdge(3, 4), 0.1);
    
    edges.push_back(makeEdge(0, 4));
    edgeCostMap = put(edgeCostMap, makeEdge(0, 4), 3.2);

    // Some more tests
    //edges.push_back(makeEdge(0, 4));
    //edgeCostMap = put(edgeCostMap, makeEdge(0, 4), 0.6);

    //edges.push_back(makeEdge(4, 2));
    //edgeCostMap = put(edgeCostMap, makeEdge(4, 2), 1.5);

    return std::make_pair(edges, edgeCostMap);
}

inline void dijkstraTest() {
    std::cout << "Dijkstra test:" << std::endl;

    auto edges_and_edge_cost_map = testEdgesWithCost();
    std::list<Edge> edges = edges_and_edge_cost_map.first;
    EdgeCostMap ecm = edges_and_edge_cost_map.second;

    Graph g(edges);
    Vertex start = 0;
    Cost start_cost = 0;
    Cost other_vertex_base_cost = 100.0;

    VertexCostMap vcm = CppDijkstraVisitor::initMap(
        CppDijkstraVisitor::vertices(g), other_vertex_base_cost);

    VertexPredecessorMap vpm;

    CppDijkstraVisitor::dijkstraShortestPaths(g, start, vcm, ecm, start_cost,
                                              vpm);

    for (auto i = 0; i < 5; ++i) {
        std::cout << "Distance to " << i << ": " << get(vcm, i) << std::endl; 
    }
}
}
