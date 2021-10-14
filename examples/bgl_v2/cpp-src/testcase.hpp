#pragma once

#include <list>
#include <random>
#include <utility>

#define NB_TEST_VERTICES 1000000
#define NB_TEST_EDGES 20000000

std::pair<int, std::list<std::pair<int, int>>> gen_test_case();
