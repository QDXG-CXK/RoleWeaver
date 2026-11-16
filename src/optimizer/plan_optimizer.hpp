#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <utility>

namespace rbac_opt {

using u64 = unsigned long long;
static constexpr double INF_D = 1e100;
static constexpr double BRANCH_CUT_TIME_LIMIT_SEC = 60.0;

struct Args {
    std::string roles_path;
    int doc_count = 0;
    double rho = 0.3;
    int M = 16;
    std::string solver = "greedy"; // branch-cut | greedy
    int exact_threshold = 160;
    long long node_limit = 2000000;
};

struct Instance {
    int n = 0;
    int original_role_count = 0;
    int doc_count = 0;
    int root = -1;
    int words = 0;
    std::vector<std::vector<int>> docs;
    std::vector<int> sizes;
    std::vector<std::vector<u64>> bits;
    std::vector<long long> S;
    std::vector<std::vector<int>> serve_cand;     // role r can be served by these selected ancestors
    std::vector<std::vector<int>> cover_roles;    // selected a covers these roles
    std::vector<std::vector<u64>> cover_mask;     // bitset over roles
    std::vector<std::vector<unsigned char>> chain; // chain[a][b] if b subset a and strict
    long long comp_edges = 0;
    long long serving_edges = 0;
    int M = 16;
};

static int popcount64(u64 x) { return (int)__builtin_popcountll(x); }

static std::vector<int> sorted_unique(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

static Instance readInstance(const Args& args, double& input_read_sec) {
    const auto read_start = std::chrono::steady_clock::now();
    std::ifstream in(args.roles_path);
    if (!in) throw std::runtime_error("cannot open role file: " + args.roles_path);
    std::vector<std::vector<int>> roles;
    int max_doc = -1;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int rid;
        if (!(ss >> rid)) continue;
        if (rid < 0) throw std::runtime_error("negative role id in role file");
        if ((int)roles.size() <= rid) roles.resize(rid + 1);
        int x;
        while (ss >> x) {
            if (x < 0) throw std::runtime_error("negative document/block id in role file");
            roles[rid].push_back(x);
            max_doc = std::max(max_doc, x);
        }
        roles[rid] = sorted_unique(std::move(roles[rid]));
    }
    if (!in.eof()) throw std::runtime_error("failed to read role file: " + args.roles_path);
    input_read_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - read_start).count();
    int doc_count = args.doc_count > 0 ? args.doc_count : (max_doc + 1);
    if (doc_count <= 0) throw std::runtime_error("cannot determine a positive base-vector count");
    for (auto& v : roles) {
        for (int x : v) {
            if (x >= doc_count) throw std::runtime_error("role file contains a vector label outside the base-vector count");
        }
    }
    int original = (int)roles.size();
    int root = -1;
    for (int i = 0; i < (int)roles.size(); ++i) {
        if ((int)roles[i].size() == doc_count) { root = i; break; }
    }
    if (root < 0) {
        root = (int)roles.size();
        roles.emplace_back();
        roles.back().resize(doc_count);
        std::iota(roles.back().begin(), roles.back().end(), 0);
    }

    Instance inst;
    inst.docs = std::move(roles);
    inst.n = (int)inst.docs.size();
    inst.original_role_count = original;
    inst.doc_count = doc_count;
    inst.root = root;
    inst.words = (doc_count + 63) >> 6;
    inst.M = args.M;
    inst.sizes.resize(inst.n);
    inst.bits.assign(inst.n, std::vector<u64>(inst.words, 0));
    inst.S.resize(inst.n);
    for (int r = 0; r < inst.n; ++r) {
        inst.sizes[r] = (int)inst.docs[r].size();
        inst.S[r] = 2LL * args.M * inst.sizes[r];
        for (int d : inst.docs[r]) inst.bits[r][d >> 6] |= 1ULL << (d & 63);
    }
    return inst;
}

static bool subsetBits(const std::vector<u64>& child, const std::vector<u64>& anc) {
    for (size_t i = 0; i < child.size(); ++i) if (child[i] & ~anc[i]) return false;
    return true;
}
static bool equalBits(const std::vector<u64>& a, const std::vector<u64>& b) {
    return a == b;
}

static void computeRelations(Instance& inst, double rho) {
    int n = inst.n;
    inst.serve_cand.assign(n, {});
    inst.cover_roles.assign(n, {});
    int role_words = (n + 63) >> 6;
    inst.cover_mask.assign(n, std::vector<u64>(role_words, 0));
    inst.chain.assign(n, std::vector<unsigned char>(n, 0));
    inst.comp_edges = 0;
    inst.serving_edges = 0;
    for (int a = 0; a < n; ++a) {
        for (int r = 0; r < n; ++r) {
            bool is_ancestor = subsetBits(inst.bits[r], inst.bits[a]);
            if (!is_ancestor) continue;
            bool can_serve = false;
            if (a == r) can_serve = true;
            else if (inst.sizes[a] > 0) {
                double ratio = (double)inst.sizes[r] / (double)inst.sizes[a];
                if (ratio + 1e-15 >= rho) can_serve = true;
            } else if (inst.sizes[r] == 0 && rho <= 1.0 + 1e-15) {
                can_serve = true;
            }
            if (can_serve) {
                inst.serve_cand[r].push_back(a);
                inst.cover_roles[a].push_back(r);
                inst.cover_mask[a][r >> 6] |= 1ULL << (r & 63);
                ++inst.serving_edges;
            }
            if (a != r && !equalBits(inst.bits[a], inst.bits[r])) {
                inst.chain[a][r] = 1;
                ++inst.comp_edges;
            }
        }
    }
    for (int r = 0; r < n; ++r) {
        std::sort(inst.serve_cand[r].begin(), inst.serve_cand[r].end(), [&](int x, int y){
            if (inst.sizes[x] != inst.sizes[y]) return inst.sizes[x] < inst.sizes[y];
            return x < y;
        });
        if (std::find(inst.serve_cand[r].begin(), inst.serve_cand[r].end(), r) == inst.serve_cand[r].end()) {
            inst.serve_cand[r].push_back(r);
        }
    }
}

struct EvalResult {
    double cost = INF_D;
    std::vector<int> selected;
    std::vector<std::pair<int,int>> edges;
    std::vector<std::vector<int>> paths;
    std::vector<int> server;
    std::vector<int> uncovered;
    int path_count = 0;
};

struct MFEdge { int to, rev, cap; long long cost; int tag_a, tag_b; };

struct MinCostMatcher {
    int N = 0, s = 0, t = 0;
    std::vector<std::vector<MFEdge>> g;
    explicit MinCostMatcher(int n=0): N(n), s(n-2), t(n-1), g(n) {}
    void addEdge(int v, int to, int cap, long long cost, int tag_a=-1, int tag_b=-1) {
        MFEdge a{to, (int)g[to].size(), cap, cost, tag_a, tag_b};
        MFEdge b{v, (int)g[v].size(), 0, -cost, -1, -1};
        g[v].push_back(a); g[to].push_back(b);
    }
    std::pair<long long, std::vector<std::pair<int,int>>> run() {
        long long total_cost = 0;
        const long long INF = (1LL << 60);
        while (true) {
            std::vector<long long> d(N, INF);
            std::vector<int> inq(N, 0), pv(N, -1), pe(N, -1);
            std::queue<int> q;
            d[s] = 0; q.push(s); inq[s] = 1;
            while (!q.empty()) {
                int v = q.front(); q.pop(); inq[v] = 0;
                for (int i = 0; i < (int)g[v].size(); ++i) {
                    const auto& e = g[v][i];
                    if (e.cap <= 0) continue;
                    if (d[e.to] > d[v] + e.cost) {
                        d[e.to] = d[v] + e.cost;
                        pv[e.to] = v; pe[e.to] = i;
                        if (!inq[e.to]) { inq[e.to] = 1; q.push(e.to); }
                    }
                }
            }
            if (d[t] >= 0 || d[t] == INF) break; // optional matching: only augment negative-saving paths
            int aug = 1;
            for (int v = t; v != s; v = pv[v]) {
                auto& e = g[pv[v]][pe[v]];
                e.cap -= aug;
                g[v][e.rev].cap += aug;
            }
            total_cost += d[t];
        }
        std::vector<std::pair<int,int>> chosen;
        for (int v = 0; v < N; ++v) {
            for (const auto& e : g[v]) {
                if (e.tag_a >= 0 && e.tag_b >= 0 && e.cap == 0) chosen.push_back({e.tag_a, e.tag_b});
            }
        }
        return {total_cost, chosen};
    }
};

static EvalResult evaluateSelected(const Instance& inst, const std::vector<unsigned char>& sel, bool need_full=true) {
    EvalResult ev;
    int n = inst.n;
    for (int i = 0; i < n; ++i) if (sel[i]) ev.selected.push_back(i);
    long long base = 0;
    for (int r : ev.selected) base += inst.S[r];
    int ssz = (int)ev.selected.size();
    MinCostMatcher mf(2 * ssz + 2);
    int Snode = 2 * ssz, Tnode = 2 * ssz + 1;
    mf.s = Snode; mf.t = Tnode;
    for (int i = 0; i < ssz; ++i) {
        mf.addEdge(Snode, i, 1, 0);
        mf.addEdge(ssz + i, Tnode, 1, 0);
    }
    for (int i = 0; i < ssz; ++i) {
        int a = ev.selected[i];
        for (int j = 0; j < ssz; ++j) {
            int b = ev.selected[j];
            if (inst.chain[a][b]) mf.addEdge(i, ssz + j, 1, -inst.S[b], a, b);
        }
    }
    auto res = mf.run();
    ev.cost = (double)(base + res.first);
    ev.edges = std::move(res.second);

    if (!need_full) return ev;

    std::vector<int> pred(n, -1), succ(n, -1);
    for (auto [a,b] : ev.edges) { succ[a] = b; pred[b] = a; }
    std::vector<unsigned char> seen(n, 0);
    std::vector<int> starts;
    for (int r : ev.selected) if (pred[r] < 0) starts.push_back(r);
    std::sort(starts.begin(), starts.end());
    for (int st : starts) {
        std::vector<int> p;
        int cur = st;
        while (cur >= 0 && cur < n && sel[cur] && !seen[cur]) {
            p.push_back(cur); seen[cur] = 1;
            cur = succ[cur];
        }
        if (!p.empty()) ev.paths.push_back(std::move(p));
    }
    for (int r : ev.selected) if (!seen[r]) ev.paths.push_back({r});
    ev.path_count = (int)ev.paths.size();

    ev.server.assign(n, -1);
    for (int r = 0; r < n; ++r) {
        int best = -1;
        for (int a : inst.serve_cand[r]) if (sel[a]) {
            if (best < 0 || std::make_pair(inst.sizes[a], a) < std::make_pair(inst.sizes[best], best)) best = a;
        }
        ev.server[r] = best;
        if (best < 0) ev.uncovered.push_back(r);
    }
    return ev;
}

static std::vector<u64> emptyRoleMask(int n) { return std::vector<u64>((n + 63) >> 6, 0); }
static bool allCovered(const std::vector<u64>& mask, int n) {
    int words = (n + 63) >> 6;
    for (int w = 0; w < words; ++w) {
        u64 want = ~0ULL;
        if (w == words - 1 && (n & 63)) want = (1ULL << (n & 63)) - 1ULL;
        if ((mask[w] & want) != want) return false;
    }
    return true;
}
static int firstUncovered(const std::vector<u64>& mask, int n) {
    int words = (n + 63) >> 6;
    for (int w = 0; w < words; ++w) {
        u64 want = ~0ULL;
        if (w == words - 1 && (n & 63)) want = (1ULL << (n & 63)) - 1ULL;
        u64 miss = (~mask[w]) & want;
        if (miss) return w * 64 + __builtin_ctzll(miss);
    }
    return -1;
}
static int popNewCoverage(const std::vector<u64>& cov, const std::vector<u64>& covered, int n) {
    int ans = 0;
    int words = (n + 63) >> 6;
    for (int w = 0; w < words; ++w) ans += popcount64(cov[w] & ~covered[w]);
    return ans;
}
static std::vector<u64> coverageOfSelected(const Instance& inst, const std::vector<unsigned char>& sel) {
    auto covered = emptyRoleMask(inst.n);
    for (int a = 0; a < inst.n; ++a) if (sel[a]) {
        for (size_t w = 0; w < covered.size(); ++w) covered[w] |= inst.cover_mask[a][w];
    }
    return covered;
}

struct SolveStats {
    std::string solver;
    std::string status;
    bool exact_finished = false;
    long long branch_nodes = 0;
    long long pruned_by_bound = 0;
    long long pruned_by_time = 0;
    double elapsed_sec = 0.0;
};

static EvalResult greedySolve(const Instance& inst) {
    int n = inst.n;
    std::vector<unsigned char> sel(n, 0);
    sel[inst.root] = 1;
    auto covered = coverageOfSelected(inst, sel);
    EvalResult cur = evaluateSelected(inst, sel, false);
    int guard = 0;
    while (!allCovered(covered, n) && guard++ <= n + 5) {
        int best = -1;
        double best_score = INF_D;
        int best_new = -1;
        double best_cost = INF_D;
        for (int c = 0; c < n; ++c) if (!sel[c]) {
            int nc = popNewCoverage(inst.cover_mask[c], covered, n);
            if (nc <= 0) continue;
            sel[c] = 1;
            double trial_cost = evaluateSelected(inst, sel, false).cost;
            sel[c] = 0;
            double delta = std::max(0.0, trial_cost - cur.cost);
            double score = delta / (double)nc;
            if (std::make_tuple(score, -nc, trial_cost, c) < std::make_tuple(best_score, -best_new, best_cost, best < 0 ? n+1 : best)) {
                best = c; best_score = score; best_new = nc; best_cost = trial_cost;
            }
        }
        if (best < 0) {
            int r = firstUncovered(covered, n);
            if (r < 0) break;
            best = r; // self is always valid
        }
        sel[best] = 1;
        for (size_t w = 0; w < covered.size(); ++w) covered[w] |= inst.cover_mask[best][w];
        cur = evaluateSelected(inst, sel, false);
    }

    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int> selected;
        for (int i = 0; i < n; ++i) if (sel[i] && i != inst.root) selected.push_back(i);
        std::sort(selected.begin(), selected.end(), [&](int a, int b){
            if (inst.S[a] != inst.S[b]) return inst.S[a] > inst.S[b];
            return a < b;
        });
        for (int c : selected) {
            sel[c] = 0;
            auto cov = coverageOfSelected(inst, sel);
            if (allCovered(cov, n)) {
                auto ev = evaluateSelected(inst, sel, false);
                if (ev.cost <= cur.cost + 1e-9) {
                    cur = ev; covered = std::move(cov); changed = true;
                    continue;
                }
            }
            sel[c] = 1;
        }
    }

    // Small local replacement search; avoid this quadratic pass for very large role sets.
    if (n <= 350) {
        bool improved = true;
        while (improved) {
            improved = false;
            double best_cost = cur.cost;
            int rem = -1, add = -1;
            for (int c = 0; c < n; ++c) if (sel[c] && c != inst.root) {
                sel[c] = 0;
                for (int d = 0; d < n; ++d) if (!sel[d]) {
                    sel[d] = 1;
                    auto cov = coverageOfSelected(inst, sel);
                    if (allCovered(cov, n)) {
                        auto ev = evaluateSelected(inst, sel, false);
                        if (ev.cost + 1e-9 < best_cost) {
                            best_cost = ev.cost; rem = c; add = d;
                        }
                    }
                    sel[d] = 0;
                }
                sel[c] = 1;
            }
            if (rem >= 0) {
                sel[rem] = 0; sel[add] = 1;
                covered = coverageOfSelected(inst, sel);
                cur = evaluateSelected(inst, sel, false);
                improved = true;
            }
        }
    }
    return evaluateSelected(inst, sel, true);
}

class BranchCutSearch {
public:
    const Instance& inst;
    double time_limit;
    long long node_limit;
    std::chrono::steady_clock::time_point t0;
    EvalResult best;
    std::vector<unsigned char> best_sel;
    SolveStats stats;
    bool stopped = false;

    BranchCutSearch(const Instance& inst_, double tl, long long nl, const EvalResult& inc)
        : inst(inst_), time_limit(tl), node_limit(nl), t0(std::chrono::steady_clock::now()), best(inc) {
        best_sel.assign(inst.n, 0);
        for (int r : best.selected) best_sel[r] = 1;
    }
    double elapsed() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
    bool shouldStop() {
        if (elapsed() >= time_limit) { stopped = true; stats.pruned_by_time++; return true; }
        if (stats.branch_nodes >= node_limit) { stopped = true; return true; }
        return false;
    }

    int chooseUncoveredRole(const std::vector<u64>& covered, const std::vector<unsigned char>& sel) {
        int n = inst.n;
        int best_r = -1, best_cnt = std::numeric_limits<int>::max();
        for (int r = 0; r < n; ++r) {
            if (covered[r >> 6] & (1ULL << (r & 63))) continue;
            int cnt = 0;
            for (int c : inst.serve_cand[r]) if (!sel[c]) ++cnt;
            if (cnt < best_cnt || (cnt == best_cnt && inst.sizes[r] > (best_r < 0 ? -1 : inst.sizes[best_r]))) {
                best_cnt = cnt; best_r = r;
            }
        }
        return best_r;
    }

    void dfs(std::vector<unsigned char>& sel, std::vector<u64>& covered) {
        if (shouldStop()) return;
        ++stats.branch_nodes;
        auto lb = evaluateSelected(inst, sel, false).cost;
        if (lb >= best.cost - 1e-9) { ++stats.pruned_by_bound; return; }
        if (allCovered(covered, inst.n)) {
            auto ev = evaluateSelected(inst, sel, true);
            if (ev.cost + 1e-9 < best.cost) {
                best = std::move(ev);
                best_sel = sel;
            }
            return;
        }
        int r = chooseUncoveredRole(covered, sel);
        if (r < 0) return;
        struct Cand { int id; double trial_cost; int newcov; long long s; };
        std::vector<Cand> cands;
        for (int c : inst.serve_cand[r]) if (!sel[c]) {
            int nc = popNewCoverage(inst.cover_mask[c], covered, inst.n);
            if (nc <= 0) continue;
            sel[c] = 1;
            double tc = evaluateSelected(inst, sel, false).cost;
            sel[c] = 0;
            if (tc < best.cost - 1e-9) cands.push_back({c, tc, nc, inst.S[c]});
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){
            double sa = a.trial_cost / std::max(1, a.newcov);
            double sb = b.trial_cost / std::max(1, b.newcov);
            if (std::fabs(sa - sb) > 1e-12) return sa < sb;
            if (a.trial_cost != b.trial_cost) return a.trial_cost < b.trial_cost;
            if (a.newcov != b.newcov) return a.newcov > b.newcov;
            return a.id < b.id;
        });
        for (const auto& cd : cands) {
            if (shouldStop()) return;
            sel[cd.id] = 1;
            auto old = covered;
            for (size_t w = 0; w < covered.size(); ++w) covered[w] |= inst.cover_mask[cd.id][w];
            dfs(sel, covered);
            covered = std::move(old);
            sel[cd.id] = 0;
        }
    }

    EvalResult run() {
        std::vector<unsigned char> sel(inst.n, 0);
        sel[inst.root] = 1;
        auto covered = coverageOfSelected(inst, sel);
        dfs(sel, covered);
        if (stats.pruned_by_time > 0) {
            throw std::runtime_error("branch-cut planning timed out (60-second limit); use --solver greedy instead");
        }
        stats.exact_finished = !stopped;
        stats.elapsed_sec = elapsed();
        return best;
    }
};

static void writePlanText(const std::string& path, const Instance& inst, const EvalResult& ev) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write plan: " + path);
    f << "DOC_COUNT " << inst.doc_count << "\n";
    f << "ROLE_COUNT " << inst.n << "\n";
    for (int r = 0; r < inst.n; ++r) {
        f << "ROLE " << r << " " << inst.docs[r].size();
        for (int d : inst.docs[r]) f << " " << d;
        f << "\n";
    }
    f << "PATH_COUNT " << ev.paths.size() << "\n";
    for (int i = 0; i < (int)ev.paths.size(); ++i) {
        f << "PATH " << i << " " << ev.paths[i].size();
        for (int r : ev.paths[i]) f << " " << r;
        f << "\n";
    }
    f << "SERVER_COUNT " << ev.server.size() << "\n";
    for (int r = 0; r < (int)ev.server.size(); ++r) f << "SERVER " << r << " " << ev.server[r] << "\n";
    if (!f) throw std::runtime_error("failed to write plan: " + path);
}

static std::string jsonVec(const std::vector<int>& v) {
    std::ostringstream ss; ss << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) ss << ","; ss << v[i]; }
    ss << "]"; return ss.str();
}
static std::string jsonPaths(const std::vector<std::vector<int>>& paths) {
    std::ostringstream ss; ss << "[";
    for (size_t i = 0; i < paths.size(); ++i) { if (i) ss << ","; ss << jsonVec(paths[i]); }
    ss << "]"; return ss.str();
}

static void writeStats(const std::string& path, const Instance& inst, const Args& args, const EvalResult& ev, const SolveStats& st) {
    long long root_cost = inst.S[inst.root];
    long long role_partition_no_comp = 0;
    for (int r = 0; r < inst.original_role_count; ++r) role_partition_no_comp += inst.S[r];
    std::vector<unsigned char> allsel(inst.n, 0);
    for (int r = 0; r < inst.original_role_count; ++r) allsel[r] = 1;
    EvalResult all_ev = evaluateSelected(inst, allsel, false);
    std::ostream* outp = &std::cout;
    std::ofstream fout;
    if (!path.empty()) { fout.open(path); if (!fout) throw std::runtime_error("cannot write stats: " + path); outp = &fout; }
    auto& out = *outp;
    out << std::fixed << std::setprecision(8);
    out << "{\n";
    out << "  \"format\": \"cpp_branch_cut_optimizer_v1\",\n";
    out << "  \"solver\": \"" << st.solver << "\",\n";
    out << "  \"status\": \"" << st.status << "\",\n";
    out << "  \"exact_finished\": " << (st.exact_finished ? "true" : "false") << ",\n";
    out << "  \"role_count_including_root\": " << inst.n << ",\n";
    out << "  \"original_role_count\": " << inst.original_role_count << ",\n";
    out << "  \"doc_count\": " << inst.doc_count << ",\n";
    out << "  \"root\": " << inst.root << ",\n";
    out << "  \"rho\": " << args.rho << ",\n";
    out << "  \"M\": " << args.M << ",\n";
    out << "  \"comparable_edges\": " << inst.comp_edges << ",\n";
    out << "  \"serving_edges\": " << inst.serving_edges << ",\n";
    out << "  \"root_cost\": " << root_cost << ",\n";
    out << "  \"role_partition_no_compression_cost\": " << role_partition_no_comp << ",\n";
    out << "  \"role_partition_no_compression_over_root\": " << ((double)role_partition_no_comp / std::max(1LL, root_cost)) << ",\n";
    out << "  \"role_partition_compressed_cost\": " << all_ev.cost << ",\n";
    out << "  \"role_partition_compressed_over_root\": " << (all_ev.cost / std::max(1LL, root_cost)) << ",\n";
    out << "  \"objective\": " << ev.cost << ",\n";
    out << "  \"exact_over_root\": " << (ev.cost / std::max(1LL, root_cost)) << ",\n";
    out << "  \"selected_count\": " << ev.selected.size() << ",\n";
    out << "  \"path_count\": " << ev.paths.size() << ",\n";
    out << "  \"selected_roles\": " << jsonVec(ev.selected) << ",\n";
    out << "  \"paths\": " << jsonPaths(ev.paths) << ",\n";
    out << "  \"uncovered_count\": " << ev.uncovered.size() << ",\n";
    out << "  \"branch_nodes\": " << st.branch_nodes << ",\n";
    out << "  \"pruned_by_bound\": " << st.pruned_by_bound << ",\n";
    out << "  \"pruned_by_time\": " << st.pruned_by_time << ",\n";
    out << "  \"elapsed_sec\": " << st.elapsed_sec << "\n";
    out << "}\n";
    if (!out) throw std::runtime_error("failed to write optimizer stats: " + path);
}

// Role-file parsing is separate so callers can report input and optimization time.
static std::pair<EvalResult, SolveStats> optimize(Instance& inst, const Args& args) {
    auto t0 = std::chrono::steady_clock::now();
    computeRelations(inst, args.rho);
    EvalResult inc = greedySolve(inst);
    SolveStats st;
    st.solver = args.solver;
    st.status = "greedy_incumbent";
    st.exact_finished = false;
    if (args.solver == "branch-cut" && inst.n <= args.exact_threshold) {
        double used = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double remain = BRANCH_CUT_TIME_LIMIT_SEC - used;
        BranchCutSearch bc(inst, remain, args.node_limit, inc);
        inc = bc.run();
        st = bc.stats;
        st.solver = "branch-cut";
        st.status = st.exact_finished ? "exact_closed" : "node_limited_incumbent";
    } else if (args.solver == "branch-cut") {
        st.solver = "branch-cut";
        st.status = "large_instance_greedy_lazy_incumbent";
    } else {
        st.solver = "greedy";
        st.status = "greedy_lazy_local_search";
    }
    st.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!inc.uncovered.empty()) {
        throw std::runtime_error("optimizer produced uncovered roles; this should not happen");
    }
    return {std::move(inc), std::move(st)};
}

} // namespace rbac_opt
