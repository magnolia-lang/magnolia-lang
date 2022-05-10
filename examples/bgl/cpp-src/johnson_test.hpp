#pragma once

#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"

namespace johnson {
// Johnson test
using examples::bgl::mg_src::bgl_cpp::CppJohnson;
typedef CppJohnson::Cost Cost;
typedef CppJohnson::Edge Edge;
typedef CppJohnson::Graph Graph;
typedef CppJohnson::Vertex Vertex;
typedef CppJohnson::VertexCostMap VertexCostMap;
typedef CppJohnson::EdgeCostMap EdgeCostMap;
typedef CppJohnson::VertexDescriptor VertexDescriptor;
typedef CppJohnson::VertexIterator VertexIterator;
typedef CppJohnson::VertexPredecessorMap VertexPredecessorMap;

auto makeEdge = CppJohnson::makeEdge;
auto get = CppJohnson::get;
auto put = CppJohnson::put;
auto toEdgeDescriptor = CppJohnson::toEdgeDescriptor;
auto toVertexDescriptor = CppJohnson::toVertexDescriptor;


inline void testGraphWithCost(Graph &g, EdgeCostMap &ecm) {
    std::list<CppJohnson::Edge> edges;
    ecm = CppJohnson::emptyECMap();

    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(0, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(1, 4));
    edges.push_back(makeEdge(2, 1));
    edges.push_back(makeEdge(3, 2));
    edges.push_back(makeEdge(3, 0));
    edges.push_back(makeEdge(4, 2));

    g = Graph(edges.begin(), edges.end(),
              boost::graph_traits<Graph>::vertices_size_type(5));

    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(1, g), g), 3.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(4, g), g), -4.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(0, g), toVertexDescriptor(2, g), g), 8.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(3, g), g), 1.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(1, g), toVertexDescriptor(4, g), g), 7.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(2, g), toVertexDescriptor(1, g), g), 4.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(3, g), toVertexDescriptor(2, g), g), -5.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(3, g), toVertexDescriptor(0, g), g), 2.0);
    put(ecm, toEdgeDescriptor(toVertexDescriptor(4, g), toVertexDescriptor(2, g), g), 6.0);
}

inline void johnsonTest() {
    std::cout << "Johnson test:" << std::endl;

    Graph g;
    EdgeCostMap ecm;
    testGraphWithCost(g, ecm);

    VertexIterator itr;
    CppJohnson::vertices(g, itr);
    CppJohnson::VertexCostMatrix vcmat = CppJohnson::emptyVCMatrix();

    for (; !CppJohnson::vertexIterEnd(itr); CppJohnson::vertexIterNext(itr)) {
        put(vcmat, CppJohnson::vertexIterUnpack(itr), CppJohnson::emptyVCMap());
    }
    CppJohnson::Unit u = CppJohnson::unit();
    CppJohnson::Bool success;

    CppJohnson::johnsonAllPairsShortestPaths(g, ecm, u, vcmat, success);

    for (auto i = 0; i < 5; ++i) {
        for (auto j = 0; j < 5; ++j) {
            std::cout << "Shortest distance from " << i << " to " << j << ": "
                      << get(get(vcmat, i), j) << std::endl;
        }
    }
}
}
