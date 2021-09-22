#pragma once

#include <chrono>
#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"
#include "testcase.hpp"

namespace dfs {
// DFS test
using examples::bgl::mg_src::bgl_cpp::CppDFSTestVisitor;
typedef CppDFSTestVisitor::Graph Graph;
typedef CppDFSTestVisitor::Vertex Vertex;
typedef CppDFSTestVisitor::VertexDescriptor VertexDescriptor;

auto makeEdge = CppDFSTestVisitor::makeEdge;
auto emptyVertexVector = CppDFSTestVisitor::emptyVertexVector;
auto toVertexDescriptor = CppDFSTestVisitor::toVertexDescriptor;

inline std::list<CppDFSTestVisitor::Edge> testEdges() {
    std::list<CppDFSTestVisitor::Edge> edges;
    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(4, 5));
    edges.push_back(makeEdge(3, 6));

    return edges;
}

inline void dfsTest() {
    std::cout << "DFS test:" << std::endl;

    auto edges = testEdges();

    Graph g(edges.begin(), edges.end(), 7);
    VertexDescriptor start = toVertexDescriptor(0, g);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto dfsResult =
        CppDFSTestVisitor::depthFirstSearch(g, start, emptyVertexVector());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    for (auto vit = dfsResult.begin(); vit != dfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

}
