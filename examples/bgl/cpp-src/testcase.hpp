#pragma once

#include <list>
#include <random>
#include <utility>

#ifndef NB_TEST_VERTICES
#define NB_TEST_VERTICES 100000
#endif

#ifndef NB_TEST_EDGES
#define NB_TEST_EDGES 1000000
#endif

std::pair<int, std::list<std::pair<int, int>>> gen_test_case();
std::list<int> gen_dijkstra_weights();
