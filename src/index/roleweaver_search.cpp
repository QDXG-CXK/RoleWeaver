#include "router.hpp"
#include "index_artifacts.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace rbac_hnsw;

struct QueryRow {
    int query_id = 0;
    int user_id = 0;
    int topk = 10;
    std::vector<int> roles;
};

struct Args {
    std::string data_path, index_dir, query_path, query_roles_path, gt_path;
    std::string out_csv, query_diagnostics_csv, exact_truth_out;
    int k = 10, max_queries = 0, tau = 4096, query_repeats = 3;
    std::vector<int> ef_s_values = {100};
};

static std::string need(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
    return argv[++i];
}

static std::vector<int> parseIntList(const std::string& text) {
    std::vector<int> values;
    std::string item;
    for (char ch : text) {
        if (ch == ',' || std::isspace(static_cast<unsigned char>(ch))) {
            if (!item.empty()) {
                values.push_back(std::stoi(item));
                item.clear();
            }
        } else {
            item.push_back(ch);
        }
    }
    if (!item.empty()) values.push_back(std::stoi(item));
    return values;
}

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--base") a.data_path = need(i, argc, argv, "--base");
        else if (s == "--index-dir") a.index_dir = need(i, argc, argv, "--index-dir");
        else if (s == "--queries") a.query_path = need(i, argc, argv, "--queries");
        else if (s == "--query-roles") a.query_roles_path = need(i, argc, argv, "--query-roles");
        else if (s == "--groundtruth") a.gt_path = need(i, argc, argv, "--groundtruth");
        else if (s == "--efs") a.ef_s_values = parseIntList(need(i, argc, argv, "--efs"));
        else if (s == "--k") a.k = std::stoi(need(i, argc, argv, "--k"));
        else if (s == "--max-queries") a.max_queries = std::stoi(need(i, argc, argv, "--max-queries"));
        else if (s == "--out-csv") a.out_csv = need(i, argc, argv, "--out-csv");
        else if (s == "--query-diagnostics-csv") a.query_diagnostics_csv = need(i, argc, argv, "--query-diagnostics-csv");
        else if (s == "--exact-truth-out") a.exact_truth_out = need(i, argc, argv, "--exact-truth-out");
        else if (s == "--tau") a.tau = std::stoi(need(i, argc, argv, "--tau"));
        else if (s == "--query-repeats") a.query_repeats = std::stoi(need(i, argc, argv, "--query-repeats"));
        else if (s == "--help") {
            std::cout << "Usage: roleweaver_search --base base.fbin --index-dir output/run/index --queries queries.fbin --query-roles query_roles.txt "
                         "[--tau 4096 --groundtruth provided_groundtruth.txt --exact-truth-out exact_groundtruth.txt --efs 100 (or 20,40,80) --query-repeats 3 --out-csv result.csv --query-diagnostics-csv diagnostics.csv --k 10 --max-queries 100]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown arg: " + s);
    }
    if (a.data_path.empty() || a.index_dir.empty() || a.query_path.empty() || a.query_roles_path.empty()) {
        throw std::runtime_error("--base --index-dir --queries --query-roles are required");
    }
    if (a.query_repeats <= 0) throw std::runtime_error("--query-repeats must be positive");
    if (a.k < 0 || a.max_queries < 0) throw std::runtime_error("--k and --max-queries must be nonnegative");
    if (a.ef_s_values.empty()) throw std::runtime_error("--efs requires at least one search width");
    if (a.tau < 0) throw std::runtime_error("--tau must be nonnegative");
    for (int ef : a.ef_s_values) if (ef <= 0) throw std::runtime_error("search widths must be positive");
    return a;
}

static void writeQueryDiagnosticsCsv(const std::string& path,
                                     const std::vector<QueryRow>& qrows,
                                     const std::vector<QueryViewDiagnostics>& diagnostics) {
    roleweaver::prepareOutputFile(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open query diagnostics csv: " + path);
    out << "query_index,query_id,user_id,target_docs,graph_views,total_view_docs,total_view_docs_per_target_docs\n";
    out << std::setprecision(17);
    for (size_t i = 0; i < diagnostics.size(); ++i) {
        const auto& d = diagnostics[i];
        out << i << "," << qrows[i].query_id << "," << qrows[i].user_id << ","
            << d.target_docs << "," << d.graph_views << "," << d.total_view_docs << ",";
        if (d.target_docs > 0) out << (double)d.total_view_docs / (double)d.target_docs;
        out << "\n";
    }
    if (!out) throw std::runtime_error("failed to write query diagnostics csv: " + path);
}

static std::vector<QueryRow> readQueryRoles(const std::string& path, int maxq) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open query roles: " + path);
    std::vector<QueryRow> rows;
    std::unordered_set<int> query_ids;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream ss(line);
        QueryRow q;
        int rc = 0;
        const auto error = [&] (const std::string& reason) {
            return std::runtime_error(path + ":" + std::to_string(line_number) + ": " + reason);
        };
        if (!(ss >> q.query_id >> q.user_id >> q.topk >> rc)) throw error("expected query_id user_id topk role_count");
        if (q.topk <= 0 || rc < 0) throw error("topk must be positive and role_count nonnegative");
        if (!query_ids.insert(q.query_id).second) throw error("duplicate query_id");
        for (int i = 0; i < rc; ++i) {
            int role;
            if (!(ss >> role) || role < 0) throw error("missing or invalid role ID");
            q.roles.push_back(role);
        }
        std::string extra;
        if (ss >> extra) throw error("unexpected fields after role IDs");
        rows.push_back(std::move(q));
        if (maxq > 0 && (int)rows.size() >= maxq) break;
    }
    return rows;
}

static std::vector<std::vector<int>> readGroundTruth(const std::string& path, int maxq,
                                                   const std::vector<QueryRow>& queries, int doc_count, int k) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open groundtruth: " + path);
    std::vector<std::vector<int>> gt;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream ss(line);
        int qid = 0, cnt = 0;
        const auto error = [&] (const std::string& reason) {
            return std::runtime_error(path + ":" + std::to_string(line_number) + ": " + reason);
        };
        if (!(ss >> qid >> cnt) || cnt < 0) throw error("expected query_id and nonnegative label_count");
        if (gt.size() >= queries.size()) throw error("more groundtruth rows than evaluated queries");
        if (qid != queries[gt.size()].query_id) throw error("query_id does not match query roles row");
        const int expected_k = k > 0 ? k : queries[gt.size()].topk;
        if (cnt > expected_k) throw error("label_count exceeds evaluated k");
        std::vector<int> labels;
        std::unordered_set<int> seen;
        for (int i = 0; i < cnt; ++i) {
            int label;
            if (!(ss >> label) || label < 0 || label >= doc_count) throw error("missing or invalid groundtruth label");
            if (!seen.insert(label).second) throw error("duplicate groundtruth label");
            labels.push_back(label);
        }
        std::string extra;
        if (ss >> extra) throw error("unexpected fields after groundtruth labels");
        gt.push_back(std::move(labels));
        if (maxq > 0 && (int)gt.size() >= maxq) break;
    }
    if (gt.size() != queries.size()) throw std::runtime_error(path + ": groundtruth row count does not match evaluated queries");
    return gt;
}

static double recall_at_k_labels(const std::vector<SearchResult>& got, const std::vector<int>& truth) {
    std::unordered_set<int> t(truth.begin(), truth.end());
    int hit = 0;
    for (auto& x : got) if (t.count(x.label)) ++hit;
    return truth.empty() ? 0.0 : (double)hit / (double)truth.size();
}

struct Accum {
    double ms = 0.0;
    double recall = 0.0;
    long long fallback = 0;
    long long graph_views = 0;
    long long hops = 0;
    long long dist = 0;
    long long raw_edges = 0;
    long long unique_edges = 0;
    long long pruned_edges = 0;
    long long removed_edges = 0;
};

struct DiagAccum {
    double assigned_roles = 0.0;
    double target_docs = 0.0;
    double cover_nodes = 0.0;
    double graph_views = 0.0;
    double scan_roles = 0.0;
    double single_graph_view = 0.0;
    double total_view_docs = 0.0;
    double avg_view_selectivity = 0.0;
    double min_view_selectivity = 0.0;
};

static void print_method(const std::string& name, const Accum& a, int qn) {
    std::cout << name << "_avg_ms," << (a.ms / std::max(1, qn)) << "\n";
    std::cout << name << "_recall," << (a.recall / std::max(1, qn)) << "\n";
    std::cout << name << "_fallback_queries," << a.fallback << "\n";
    std::cout << name << "_avg_graph_views," << ((double)a.graph_views / std::max(1, qn)) << "\n";
    std::cout << name << "_avg_hops," << ((double)a.hops / std::max(1, qn)) << "\n";
    std::cout << name << "_avg_dist_computations," << ((double)a.dist / std::max(1, qn)) << "\n";
    std::cout << name << "_avg_raw_edges," << ((double)a.raw_edges / std::max(1, qn)) << "\n";
    std::cout << name << "_avg_unique_edges," << ((double)a.unique_edges / std::max(1, qn)) << "\n";
    std::cout << name << "_avg_post_prune_edges," << ((double)a.pruned_edges / std::max(1, qn)) << "\n";
    std::cout << name << "_avg_removed_edges," << ((double)a.removed_edges / std::max(1, qn)) << "\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        int n = 0, dim = 0, qn0 = 0, qdim = 0;
        std::cerr << "[search] loading data: " << args.data_path << "\n";
        auto data = readFbin(args.data_path, n, dim);
        std::cerr << "[search] loading queries: " << args.query_path << "\n";
        auto qdata = readFbin(args.query_path, qn0, qdim);
        if (qdim != dim) throw std::runtime_error("query dim != base dim");
        std::cerr << "[search] loading query roles: " << args.query_roles_path << "\n";
        auto qrows = readQueryRoles(args.query_roles_path, args.max_queries);
        const int expected_queries = args.max_queries > 0 ? std::min(args.max_queries, qn0) : qn0;
        if ((int)qrows.size() != expected_queries) throw std::runtime_error("query roles row count does not match query vectors (and --max-queries)");
        int qn = (int)qrows.size();
        auto manifest = roleweaver::readManifest(args.index_dir);
        roleweaver::validateArtifacts(args.index_dir, manifest, data, n, dim, args.data_path);
        auto plan = readPlanText(roleweaver::artifactPath(args.index_dir, "plan.txt"));
        if (plan.doc_count != n) throw std::runtime_error("plan vector count does not match base vectors");
        const int M = roleweaver::positiveInt(manifest, "M");
        auto load_start = std::chrono::steady_clock::now();
        RolePathIndex index(std::move(data), n, dim, std::move(plan), M,
                            roleweaver::positiveInt(manifest, "ef_c"),
                            roleweaver::positiveInt(manifest, "build_threads"),
                            roleweaver::positiveInt(manifest, "max_path_batch"));
        index.loadBinary(roleweaver::artifactPath(args.index_dir, "index.bin"));
        const double index_load_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_start).count();
        const uint64_t index_size_bytes = index.serializedSizeBytes();
        std::cerr << "[search] loaded index in " << index_load_sec << " sec\n";

        // Router diagnostics are deliberately computed outside both the index build/load
        // timers and the per-query search timers. Reuse them for every ef_s sweep.
        std::vector<QueryViewDiagnostics> query_diagnostics;
        query_diagnostics.reserve(qn);
        for (const auto& row : qrows) {
            query_diagnostics.push_back(index.diagnoseQueryViews(row.roles));
        }
        std::vector<std::vector<int>> groundtruth;
        bool loaded_truth = false;
        double truth_generation_ms = 0.0;
        if (!args.gt_path.empty()) {
            try {
                groundtruth = readGroundTruth(args.gt_path, args.max_queries > 0 ? qn : 0, qrows, n, args.k);
                for (int i = 0; i < qn; ++i) {
                    const int k = args.k > 0 ? args.k : qrows[i].topk;
                    if (static_cast<int>(groundtruth[i].size()) != std::min(k, query_diagnostics[i].target_docs))
                        throw std::runtime_error("groundtruth label_count does not match authorized top-k for query_id " + std::to_string(qrows[i].query_id));
                }
                loaded_truth = true;
                std::cerr << "[search] loaded ground truth: " << args.gt_path << "\n";
            } catch (const std::runtime_error& error) {
                groundtruth.clear();
                std::cerr << "[search] cannot use ground truth: " << error.what()
                          << "; generating exact ground truth\n";
            }
        }
        if (!loaded_truth) {
            std::cerr << "[search] computing exact ground truth for " << qn << " queries\n";
            groundtruth.reserve(qn);
            for (int i = 0; i < qn; ++i) {
                const int k = args.k > 0 ? args.k : qrows[i].topk;
                const auto start = std::chrono::steady_clock::now();
                const auto exact = index.bruteForceRolesUnion(qrows[i].roles, qdata.data() + static_cast<size_t>(i) * dim, k);
                truth_generation_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                groundtruth.emplace_back();
                for (const auto& result : exact) groundtruth.back().push_back(result.label);
            }
        }
        if (!args.exact_truth_out.empty()) {
            roleweaver::prepareOutputFile(args.exact_truth_out);
            std::ofstream out(args.exact_truth_out);
            if (!out) throw std::runtime_error("cannot open ground truth output: " + args.exact_truth_out);
            for (int i = 0; i < qn; ++i) {
                out << qrows[i].query_id << " " << groundtruth[i].size();
                for (int label : groundtruth[i]) out << " " << label;
                out << "\n";
            }
            if (!out) throw std::runtime_error("failed to write ground truth output: " + args.exact_truth_out);
        }

        RouterOptions options;
        options.tau = args.tau;
        // Warm up the actual router with the evaluated k and search widths.
        for (int ef_s : args.ef_s_values) {
            for (int i = 0; i < std::min(qn, 5); ++i) {
                const int k = args.k > 0 ? args.k : qrows[i].topk;
                queryRouter(index, qrows[i].roles, qdata.data() + static_cast<size_t>(i) * dim,
                            k, ef_s, query_diagnostics[i], options);
            }
        }

        auto run_once = [&](int ef_s, bool print_detail) {
        Accum aro;
        DiagAccum diag;
        double avg_cover = 0.0;
        int max_cover = 0;
        std::cerr << "[search] running " << qn << " queries\n";
        for (int i = 0; i < qn; ++i) {
            if (i > 0 && (i % 50 == 0 || i == qn - 1)) {
                std::cerr << "[search] queries done: " << i << "/" << qn << "\n";
            }
            const float* q = qdata.data() + (size_t)i * dim;
            int k = args.k > 0 ? args.k : qrows[i].topk;
            std::vector<int> cover = index.exactCoverRoles(qrows[i].roles);
            avg_cover += cover.size();
            max_cover = std::max(max_cover, (int)cover.size());
            const auto& qdiag = query_diagnostics[i];
            diag.assigned_roles += qdiag.assigned_roles;
            diag.target_docs += qdiag.target_docs;
            diag.cover_nodes += qdiag.cover_nodes;
            diag.graph_views += qdiag.graph_views;
            diag.scan_roles += qdiag.scan_roles;
            diag.single_graph_view += qdiag.single_graph_view;
            diag.total_view_docs += qdiag.total_view_docs;
            diag.avg_view_selectivity += qdiag.avg_view_selectivity;
            diag.min_view_selectivity += qdiag.min_view_selectivity;

            {
                std::vector<SearchResult> ans;
                auto t0 = std::chrono::steady_clock::now();
                RouterRoute route;
                ans = queryRouter(index, qrows[i].roles, q, k, ef_s, qdiag, options, &route);
                aro.fallback += route == RouterRoute::Scan;
                aro.graph_views += route == RouterRoute::Independent;
                auto t1 = std::chrono::steady_clock::now();
                aro.ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                aro.recall += recall_at_k_labels(ans, groundtruth[i]);
            }
        }
        std::cerr << "[search] queries done: " << qn << "/" << qn << "\n";

        if (print_detail) {
            std::cout << "index_load_sec," << index_load_sec << "\n";
            std::cout << "index_size_bytes," << index_size_bytes << "\n";
            std::cout << "docs," << index.docCount() << "\n";
            std::cout << "roles," << index.roleCount() << "\n";
            std::cout << "dim," << index.dim() << "\n";
            std::cout << "distance_kernel," << simd::kernel_name() << "\n";
            std::cout << "queries," << qn << "\n";
            std::cout << "ef_s," << ef_s << "\n";
            std::cout << "k," << args.k << "\n";
            std::cout << "avg_truth_generation_ms," << (truth_generation_ms / std::max(1, qn)) << "\n";
            std::cout << "avg_cover_nodes," << (avg_cover / std::max(1, qn)) << "\n";
            std::cout << "max_cover_nodes," << max_cover << "\n";
            std::cout << "cover_cache_entries," << index.coverCacheSize() << "\n";
            std::cout << "diag_avg_assigned_roles," << (diag.assigned_roles / std::max(1, qn)) << "\n";
            std::cout << "diag_avg_target_docs," << (diag.target_docs / std::max(1, qn)) << "\n";
            std::cout << "diag_avg_graph_views," << (diag.graph_views / std::max(1, qn)) << "\n";
            std::cout << "diag_avg_scan_roles," << (diag.scan_roles / std::max(1, qn)) << "\n";
            std::cout << "diag_single_graph_view_fraction," << (diag.single_graph_view / std::max(1, qn)) << "\n";
            std::cout << "diag_avg_total_view_docs," << (diag.total_view_docs / std::max(1, qn)) << "\n";
            std::cout << "diag_avg_view_selectivity," << (diag.avg_view_selectivity / std::max(1, qn)) << "\n";
            std::cout << "diag_avg_min_view_selectivity," << (diag.min_view_selectivity / std::max(1, qn)) << "\n";
            std::cout << "groundtruth_source," << (loaded_truth ? "loaded" : "generated") << "\n";
            print_method("router", aro, qn);
        }
        double ms = aro.ms;
        double recall_sum = aro.recall;
        double qps = ms > 0.0 ? 1000.0 * (double)qn / ms : 0.0;
        double recall = recall_sum / std::max(1, qn);
        return std::make_pair(qps, recall);
        };

        if (!args.out_csv.empty()) {
            roleweaver::prepareOutputFile(args.out_csv);
            std::ofstream out(args.out_csv);
            if (!out) throw std::runtime_error("cannot open output csv: " + args.out_csv);
            for (int ef_s : args.ef_s_values) {
                double qps_sum = 0.0;
                double recall_sum = 0.0;
                for (int repeat = 0; repeat < args.query_repeats; ++repeat) {
                    std::cerr << "[search] ef_s=" << ef_s << " repeat=" << (repeat + 1)
                              << "/" << args.query_repeats << "\n";
                    auto result = run_once(ef_s, false);
                    qps_sum += result.first;
                    recall_sum += result.second;
                }
                out << ef_s << "," << (qps_sum / args.query_repeats)
                    << "," << (recall_sum / args.query_repeats) << "\n";
            }
            std::cerr << "[search] wrote sweep csv: " << args.out_csv << "\n";
        } else {
            for (int ef_s : args.ef_s_values) {
                double qps = 0.0, recall = 0.0;
                for (int repeat = 0; repeat < args.query_repeats; ++repeat) {
                    const auto result = run_once(ef_s, repeat == 0);
                    qps += result.first;
                    recall += result.second;
                }
                std::cout << "qps," << qps / args.query_repeats << "\n"
                          << "recall," << recall / args.query_repeats << "\n";
            }
        }
        if (!args.query_diagnostics_csv.empty()) {
            writeQueryDiagnosticsCsv(args.query_diagnostics_csv, qrows, query_diagnostics);
            std::cerr << "[search] wrote query diagnostics csv: " << args.query_diagnostics_csv << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
