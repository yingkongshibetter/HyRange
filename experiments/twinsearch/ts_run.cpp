// Thin command-line driver around the authors' TwinSearch class (github.com/tlarock/twin_search).
// Written for the experiments's comparison; NOT part of the TwinSearch repository (MIT, like this repository).  It only (1) reads a hypergraph, (2) builds the
// authors' ProjectedGraph (with the diagonal when use_diag = 1) and TwinSearch objects, (3) calls the authors'
// TwinSearch::parallel_search, (4) prints what the search returned.
// usage: ts_run <hyperedge file: one hyperedge per line, comma separated node ids 0..n-1>
//               <min_k> <max_k (0 = n)> <use_diag 0|1> <filter_isomorphic 0|1> <threads> <drain seconds (0 = none)>
//               <twins output file or -> <max twins to dump>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <oneapi/tbb/global_control.h>
#include "hypergraph.hpp"
#include "projected_graph.hpp"
#include "factor_graph.hpp"
#include "twin_search.hpp"

static std::string he_str(const std::vector<int> &he) {
    std::string s;
    for (std::size_t i = 0; i < he.size(); i++) { if (i) s += "-"; s += std::to_string(he[i]); }
    return s;
}

int main(int argc, char **argv) {
    if (argc < 10) { std::cerr << "usage: see ts_run.cpp" << std::endl; return 2; }
    const std::string infile = argv[1];
    const int min_k = std::atoi(argv[2]);
    int max_k = std::atoi(argv[3]);
    const bool use_diag = std::atoi(argv[4]) != 0;
    const bool filter_iso = std::atoi(argv[5]) != 0;
    const int threads = std::atoi(argv[6]);
    const double drain_s = std::atof(argv[7]);
    const std::string twins_out = argv[8];
    const std::size_t max_dump = static_cast<std::size_t>(std::atoll(argv[9]));

    std::vector<std::vector<int> > hyperedges;
    {
        std::ifstream in(infile);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::vector<int> he;
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ',')) he.push_back(std::stoi(tok));
            hyperedges.push_back(he);
        }
    }
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, threads);
    const auto t0 = std::chrono::steady_clock::now();
    Hypergraph H(hyperedges);
    if (max_k <= 0) max_k = H.n;
    ProjectedGraph proj(H, !use_diag);          // zero_diagonal = false keeps the node degrees on the diagonal
    // filter_isomorphic, parallel = true, run_search = false (search is called below so that it can be timed)
    TwinSearch ts(proj, min_k, max_k, filter_iso, true, false, use_diag);
    const auto t1 = std::chrono::steady_clock::now();
    if (drain_s > 0) ts.arm_drain(drain_s);
    if (ts.feasible) ts.parallel_search(filter_iso);
    const auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0; };

    // hyperedges present in every listed hypergraph
    std::map<int, std::size_t> cnt;
    for (const std::vector<int> &tw : ts.twins) for (int c : tw) cnt[c]++;
    std::vector<std::string> common;
    for (const auto &[c, k] : cnt) if (k == ts.twins.size()) common.push_back(he_str(ts.fact.node_map.at(c)));

    bool dumped = false;
    if (twins_out != "-" && !ts.drained && ts.twins.size() <= max_dump) {
        std::ofstream out(twins_out, std::ios::out | std::ios::trunc);
        for (const std::vector<int> &tw : ts.twins) {
            std::vector<std::vector<int> > hes = ts.inflate_cnodes(tw);
            for (std::size_t i = 0; i < hes.size(); i++) { if (i) out << "|"; out << he_str(hes[i]); }
            out << "\n";
        }
        dumped = true;
    }
    std::cout << "{\"n\":" << H.n << ",\"m\":" << H.m << ",\"remapped\":" << (H.nodes_were_remapped ? "true" : "false")
              << ",\"min_k\":" << min_k << ",\"max_k\":" << max_k << ",\"use_diag\":" << (use_diag ? "true" : "false")
              << ",\"threads\":" << threads << ",\"feasible\":" << (ts.feasible ? "true" : "false")
              << ",\"edge_nodes\":" << ts.fact.num_edge_nodes << ",\"clique_nodes\":" << ts.fact.num_clique_nodes
              << ",\"drained\":" << (ts.drained ? "true" : "false") << ",\"twins\":" << ts.twins.size()
              << ",\"filtered_twins\":" << ts.filtered_twins.size() << ",\"mate_pairs\":" << ts.mates.size()
              << ",\"ms_setup\":" << ms(t0, t1) << ",\"ms_search\":" << ms(t1, t2)
              << ",\"ms_traversal\":" << ts.ms_traversal << ",\"ms_mates\":" << ts.ms_mates << ",\"ms_iso\":" << ts.ms_iso
              << ",\"dumped\":" << (dumped ? "true" : "false") << ",\"common\":[";
    for (std::size_t i = 0; i < common.size(); i++) { if (i) std::cout << ","; std::cout << "\"" << common[i] << "\""; }
    std::cout << "]}" << std::endl;
    return 0;
}
