#pragma once

#include <chrono>
#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"
#include "testcase.hpp"

namespace bfsParallel {
// BFS test
using examples::bgl::mg_src::bgl_cpp::CppParallelBFSTestVisitor;
typedef CppParallelBFSTestVisitor::Graph Graph;
typedef CppParallelBFSTestVisitor::Vertex Vertex;
typedef CppParallelBFSTestVisitor::VertexDescriptor VertexDescriptor;

auto makeEdge = CppParallelBFSTestVisitor::makeEdge;
auto emptyVertexVector = CppParallelBFSTestVisitor::emptyVertexVector;
auto toVertexDescriptor = CppParallelBFSTestVisitor::toVertexDescriptor;

inline std::list<CppParallelBFSTestVisitor::Edge> testEdges() {
    std::list<CppParallelBFSTestVisitor::Edge> edges;
    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(4, 5));
    edges.push_back(makeEdge(3, 6));

    return edges;
}

inline std::list<CppParallelBFSTestVisitor::Edge> lotsOfEdges() {
    auto test_case = gen_test_case();
    std::list<CppParallelBFSTestVisitor::Edge> edges;

    for (auto pair_it = test_case.second.begin();
         pair_it != test_case.second.end();
         ++pair_it) {
        edges.push_back(makeEdge(pair_it->first, pair_it->second));
    }

    return edges;
}

inline void bfsParallelTest() {
    std::cout << "BFS parallel test:" << std::endl;

    auto edges = testEdges();

    Graph g(edges.begin(), edges.end(), 7);
    VertexDescriptor start = toVertexDescriptor(0, g);

    auto bfsResult =
        CppParallelBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    for (auto vit = bfsResult.m_vec.begin(); vit != bfsResult.m_vec.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;
}

inline void bfsParallelPerfTest() {
    std::cout << "BFS parallel perf test:" << std::endl;
    auto edges = lotsOfEdges();
    Graph g(edges.begin(), edges.end(), NB_TEST_VERTICES);
    VertexDescriptor start = toVertexDescriptor(0, g);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto bfsResult =
        CppParallelBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    for (auto vit = bfsResult.m_vec.begin(); vit != bfsResult.m_vec.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}


}
