#include <iostream>

#include "testcase.hpp"

std::pair<int, std::list<std::pair<int, int>>> gen_test_case() {
    int nb_vertices = NB_TEST_VERTICES, nb_edges = NB_TEST_EDGES;
    static std::list<std::pair<int, int>> edges;

    if (edges.size() > 0) { return std::make_pair(nb_vertices, edges); }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type>
        range_obj(0, nb_vertices - 1);

    for (auto i = 0; i < nb_edges; ++i) {
        edges.push_back(std::make_pair(range_obj(rng), range_obj(rng)));
        //std::cout << "edge: " << edges.back().first << " " << edges.back().second
        //          << std::endl;
    }
    return std::make_pair(nb_vertices, edges);
}

/*
std::pair<int, std::list<std::pair<int, int>>> gen_test_case() {
    int nb_vertices = NB_TEST_VERTICES, nb_edges = NB_TEST_EDGES;
    static std::list<std::pair<int, int>> edges;

    if (edges.size() > 0) { return std::make_pair(nb_vertices, edges); }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type>
        range_obj(0, nb_vertices - 1);

    for (auto i = 0; i < nb_vertices; ++i) {
        edges.push_back(std::make_pair(0, i)); //range_obj(rng)));
        //std::cout << "edge: " << edges.back().first << " " << edges.back().second
        //          << std::endl;
    }
    return std::make_pair(nb_vertices, edges);
}*/
