#pragma once
#include "versioned_hnsw.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rbac_hnsw {

#ifdef ROLEWEAVER_LATENCY_PROFILE
// Opt-in measurement only; absent from normal benchmark builds.
struct MergePruneProfile {
    bool enabled = false;
    double merge_ns = 0, prune_ns = 0, result_merge_ns = 0;
    uint64_t intervals = 0;
};
inline thread_local MergePruneProfile merge_prune_profile;
struct ProfileInterval {
    double* accumulator;
    std::chrono::steady_clock::time_point begin;
    explicit ProfileInterval(double& value) : accumulator(merge_prune_profile.enabled ? &value : nullptr) {
        if (accumulator) begin = std::chrono::steady_clock::now();
    }
    ~ProfileInterval() {
        if (accumulator) {
            auto end = std::chrono::steady_clock::now();
            *accumulator += std::chrono::duration<double, std::nano>(end - begin).count();
            ++merge_prune_profile.intervals;
        }
    }
};
#endif

// ROLE_PATH_INDEX_VIRTUAL_MERGE_V3

struct StampFilter final : public LabelFilter {
    const std::vector<int>* stamp = nullptr;
    int tag = 0;
    StampFilter(const std::vector<int>* s, int t) : stamp(s), tag(t) {}
    bool allowed(int label) const override {
        return label >= 0 && label < (int)stamp->size() && (*stamp)[label] == tag;
    }
};

struct PlanData {
    int doc_count = 0;
    std::vector<std::vector<int>> role_docs;
    std::vector<std::vector<int>> paths_head_to_tail;
    std::vector<int> server;
};

struct VirtualSearchStats {
    int cover_nodes = 0;
    int graph_views = 0;
    int seed_count = 0;
    int hops = 0;
    int distance_computations = 0;
    int prune_distance_computations = 0;
    int raw_edges = 0;
    int unique_edges = 0;
    int pruned_edges = 0;          // Backward-compatible: number of edges kept after online prune.
    int post_prune_edges = 0;
    int removed_edges = 0;
    int independent_searches_avoided = 0;
    bool fallback_exact = false;
};

struct QueryViewDiagnostics {
    int assigned_roles = 0;
    int target_docs = 0;
    int cover_nodes = 0;
    int graph_views = 0;
    int scan_roles = 0;
    int single_graph_view = 0;
    int total_view_docs = 0;
    double avg_view_selectivity = 0.0;
    double min_view_selectivity = 1.0;
};

inline PlanData readPlanText(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open plan: " + path);
    PlanData p;
    std::string tag;
    int role_count = 0, path_count = 0, server_count = 0;
    in >> tag >> p.doc_count;
    if (tag != "DOC_COUNT") throw std::runtime_error("bad plan: expected DOC_COUNT");
    in >> tag >> role_count;
    if (tag != "ROLE_COUNT") throw std::runtime_error("bad plan: expected ROLE_COUNT");
    p.role_docs.assign(role_count, {});
    for (int i = 0; i < role_count; ++i) {
        int id, sz;
        in >> tag >> id >> sz;
        if (tag != "ROLE" || id < 0 || id >= role_count) throw std::runtime_error("bad ROLE row");
        p.role_docs[id].resize(sz);
        for (int j = 0; j < sz; ++j) in >> p.role_docs[id][j];
    }
    in >> tag >> path_count;
    if (tag != "PATH_COUNT") throw std::runtime_error("bad plan: expected PATH_COUNT");
    p.paths_head_to_tail.assign(path_count, {});
    for (int i = 0; i < path_count; ++i) {
        int id, len;
        in >> tag >> id >> len;
        if (tag != "PATH" || id < 0 || id >= path_count) throw std::runtime_error("bad PATH row");
        p.paths_head_to_tail[id].resize(len);
        for (int j = 0; j < len; ++j) in >> p.paths_head_to_tail[id][j];
    }
    in >> tag >> server_count;
    if (tag != "SERVER_COUNT") throw std::runtime_error("bad plan: expected SERVER_COUNT");
    p.server.assign(server_count, -1);
    for (int i = 0; i < server_count; ++i) {
        int id, s;
        in >> tag >> id >> s;
        if (tag != "SERVER" || id < 0 || id >= server_count) throw std::runtime_error("bad SERVER row");
        p.server[id] = s;
    }
    return p;
}

inline std::vector<float> readFbin(const std::string& path, int& n, int& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open fbin: " + path);
    int32_t ni = 0, di = 0;
    in.read(reinterpret_cast<char*>(&ni), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&di), sizeof(int32_t));
    if (ni <= 0 || di <= 0) throw std::runtime_error("bad fbin header");
    n = ni; dim = di;
    std::vector<float> data((size_t)n * dim);
    in.read(reinterpret_cast<char*>(data.data()), sizeof(float) * data.size());
    if (!in) throw std::runtime_error("bad fbin payload");
    return data;
}

inline void writeFbin(const std::string& path, const std::vector<float>& data, int n, int dim) {
    std::ofstream out(path, std::ios::binary);
    int32_t ni = n, di = dim;
    out.write(reinterpret_cast<const char*>(&ni), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&di), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(data.data()), sizeof(float) * data.size());
}

class BuildThreadPool {
public:
    using Workspace = VersionedHNSW::BuildVisitWorkspace;
    using Task = std::function<void(Workspace&)>;

    explicit BuildThreadPool(int thread_count) : thread_count_(thread_count) {
        if (thread_count_ <= 0) throw std::runtime_error("build thread count must be positive");
        workers_.reserve((size_t)thread_count_);
        for (int i = 0; i < thread_count_; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    BuildThreadPool(const BuildThreadPool&) = delete;
    BuildThreadPool& operator=(const BuildThreadPool&) = delete;

    ~BuildThreadPool() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }

    void enqueue(Task task) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (stopping_) throw std::runtime_error("cannot enqueue on a stopped build thread pool");
            tasks_.push_back(std::move(task));
        }
        ready_.notify_one();
    }

    int peakRunning() const { return peak_running_.load(std::memory_order_relaxed); }
    double busySeconds() const {
        return (double)busy_nanoseconds_.load(std::memory_order_relaxed) / 1.0e9;
    }

private:
    int thread_count_ = 0;
    std::vector<std::thread> workers_;
    std::deque<Task> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    bool stopping_ = false;
    std::atomic<int> running_{0};
    std::atomic<int> peak_running_{0};
    std::atomic<long long> busy_nanoseconds_{0};

    void workerLoop() {
        Workspace workspace;
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            const int now_running = running_.fetch_add(1, std::memory_order_relaxed) + 1;
            int peak = peak_running_.load(std::memory_order_relaxed);
            while (now_running > peak &&
                   !peak_running_.compare_exchange_weak(peak, now_running, std::memory_order_relaxed)) {}
            const auto start = std::chrono::steady_clock::now();
            task(workspace);
            const auto stop = std::chrono::steady_clock::now();
            busy_nanoseconds_.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count(),
                std::memory_order_relaxed);
            running_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
};

class RolePathIndex {
public:
    RolePathIndex(std::vector<float> data, int n, int dim, PlanData plan, int M = 16, int ef_c = 100,
                  int build_threads = (int)std::max(1u, std::thread::hardware_concurrency()),
                  int max_path_batch = 8)
        : data_(std::move(data)), n_(n), dim_(dim), plan_(std::move(plan)), M_(M), ef_c_(ef_c),
          build_threads_(build_threads), max_path_batch_(max_path_batch),
          stamp_(n, 0), seen_stamp_(n, 0), visit_stamp_(n, 0), neighbor_stamp_(n, 0),
          tmp_dist_(n, std::numeric_limits<float>::infinity()) {
        if (plan_.doc_count != n_) std::cerr << "[warn] plan doc_count=" << plan_.doc_count << " but vectors=" << n_ << "\n";
        role_path_.assign(plan_.role_docs.size(), -1);
        role_stage_.assign(plan_.role_docs.size(), -1);
        buildRoleBitsets();
    }

    void build() {
        if (build_threads_ <= 0) throw std::runtime_error("build_threads must be positive");
        if (max_path_batch_ <= 0) throw std::runtime_error("max_path_batch must be positive");
        constexpr int kBasePathBatch = 4;
        const int base_batch = std::max(1, std::min({kBasePathBatch, build_threads_, max_path_batch_}));
        const int path_count = (int)plan_.paths_head_to_tail.size();

        struct PathBuildState {
            std::vector<std::vector<int>> stage_docs;
            size_t unique_docs = 0;
            int stage = 0;
            size_t offset = 0;
            int pending = 0;
            bool active = false;
            bool finished = false;
            VersionedHNSW::PreparedBuildBatch batch;
            std::map<int, uint64_t> batch_histogram;
            std::chrono::steady_clock::time_point started_at;
        };

        indexes_.clear();
        indexes_.resize((size_t)path_count);
        role_path_.assign(plan_.role_docs.size(), -1);
        role_stage_.assign(plan_.role_docs.size(), -1);
        std::vector<PathBuildState> states((size_t)path_count);
        std::vector<int> inserted(n_, -1);

        // Precompute every path before starting workers. This removes shared role-map
        // writes from the parallel region and gives the dispatcher exact path weights.
        for (int pid = 0; pid < path_count; ++pid) {
            const auto& path = plan_.paths_head_to_tail[pid];
            auto& state = states[pid];
            state.stage_docs.resize(path.size());
            for (int rev = (int)path.size() - 1, stage = 0; rev >= 0; --rev, ++stage) {
                int role = path[rev];
                if (role < 0 || role >= roleCount()) throw std::runtime_error("bad role id in path");
                role_path_[role] = pid;
                role_stage_[role] = stage;
                for (int doc : plan_.role_docs[role]) {
                    if (doc < 0 || doc >= n_) continue;
                    if (inserted[doc] != pid) {
                        state.stage_docs[stage].push_back(doc);
                        inserted[doc] = pid;
                    }
                }
            }
            for (const auto& docs : state.stage_docs) state.unique_docs += docs.size();
            indexes_[pid] = VersionedHNSW(&data_, n_, dim_, M_, ef_c_, 1009 + pid);
        }

        const auto build_t0 = std::chrono::steady_clock::now();
        std::cerr << "[index-build] scheduler build_threads=" << build_threads_
                  << " base_path_batch=" << base_batch
                  << " max_path_batch=" << max_path_batch_ << "\n";

        // One worker is deliberately identical to the original serial insertion order.
        if (build_threads_ == 1) {
            for (int pid = 0; pid < path_count; ++pid) {
                auto& state = states[pid];
                const auto& path = plan_.paths_head_to_tail[pid];
                const auto started = std::chrono::steady_clock::now();
                std::cerr << "[index-build] path " << (pid + 1) << "/" << path_count
                          << " start roles=" << path.size() << " unique_docs=" << state.unique_docs << "\n";
                for (int stage = 0; stage < (int)state.stage_docs.size(); ++stage) {
                    for (int label : state.stage_docs[stage]) indexes_[pid].addPoint(label, stage);
                }
                indexes_[pid].finalizeLayout((int)path.size());
                const double path_sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                const uint64_t path_bytes = indexes_[pid].serializedSizeBytes();
                std::cerr << "[index-build] path " << (pid + 1) << "/" << path_count
                          << " done in " << path_sec << " sec size_bytes=" << path_bytes
                          << " size_mib=" << ((double)path_bytes / (1024.0 * 1024.0))
                          << " batches={1:" << state.unique_docs << "}\n";
            }
        } else {
            struct Completion {
                int pid = -1;
                std::exception_ptr error;
            };
            std::mutex completion_mutex;
            std::condition_variable completion_ready;
            std::deque<Completion> completions;
            std::atomic<bool> cancelled{false};
            size_t inflight = 0;
            std::exception_ptr first_error;

            std::vector<int> start_order((size_t)path_count);
            std::iota(start_order.begin(), start_order.end(), 0);
            std::stable_sort(start_order.begin(), start_order.end(), [&](int a, int b) {
                return states[a].unique_docs > states[b].unique_docs;
            });
            size_t next_to_start = 0;
            int active_paths = 0;
            int finished_paths = 0;
            int peak_active_paths = 0;
            const int desired_active = std::min(path_count,
                (build_threads_ + base_batch - 1) / base_batch);

            BuildThreadPool pool(build_threads_);

            auto logPathDone = [&](int pid) {
                auto& state = states[pid];
                indexes_[pid].finalizeLayout((int)plan_.paths_head_to_tail[pid].size());
                const double path_sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - state.started_at).count();
                const uint64_t path_bytes = indexes_[pid].serializedSizeBytes();
                std::ostringstream batches;
                batches << "{";
                bool first = true;
                for (const auto& item : state.batch_histogram) {
                    if (!first) batches << ",";
                    first = false;
                    batches << item.first << ":" << item.second;
                }
                batches << "}";
                std::cerr << "[index-build] path " << (pid + 1) << "/" << path_count
                          << " done in " << path_sec << " sec size_bytes=" << path_bytes
                          << " size_mib=" << ((double)path_bytes / (1024.0 * 1024.0))
                          << " batches=" << batches.str() << "\n";
            };

            auto finishPath = [&](int pid) {
                auto& state = states[pid];
                if (state.finished) return;
                indexes_[pid].finishConcurrentBuild();
                state.active = false;
                state.finished = true;
                --active_paths;
                ++finished_paths;
                logPathDone(pid);
            };

            std::function<void(int)> scheduleNextBatch;
            scheduleNextBatch = [&](int pid) {
                auto& state = states[pid];
                try {
                    while (state.stage < (int)state.stage_docs.size() &&
                           state.offset >= state.stage_docs[state.stage].size()) {
                        ++state.stage;
                        state.offset = 0;
                    }
                    if (state.stage >= (int)state.stage_docs.size()) {
                        finishPath(pid);
                        return;
                    }

                    int next_batch = base_batch;
                    if (next_to_start >= start_order.size() && active_paths > 0) {
                        const int fair_share = (build_threads_ + active_paths - 1) / active_paths;
                        next_batch = std::max(base_batch, fair_share);
                    }
                    next_batch = std::min({next_batch, max_path_batch_, build_threads_});
                    auto& docs = state.stage_docs[state.stage];
                    state.batch = indexes_[pid].prepareBuildBatch(
                        docs, state.offset, (size_t)next_batch, state.stage);
                    state.offset += state.batch.consumed;
                    if (state.batch.consumed == 0) {
                        throw std::runtime_error("build scheduler made no progress");
                    }
                    if (!state.batch.internals.empty()) {
                        state.batch_histogram[(int)state.batch.internals.size()]++;
                    }
                    state.pending = (int)state.batch.internals.size();
                    if (state.pending == 0) {
                        scheduleNextBatch(pid);
                        return;
                    }

                    for (size_t position = 0; position < state.batch.internals.size(); ++position) {
                        pool.enqueue([&, pid, position](VersionedHNSW::BuildVisitWorkspace& workspace) {
                            Completion completion;
                            completion.pid = pid;
                            if (!cancelled.load(std::memory_order_relaxed)) {
                                try {
                                    indexes_[pid].connectPreparedPoint(states[pid].batch, position, workspace);
                                } catch (...) {
                                    completion.error = std::current_exception();
                                    cancelled.store(true, std::memory_order_relaxed);
                                }
                            }
                            {
                                std::lock_guard<std::mutex> guard(completion_mutex);
                                completions.push_back(std::move(completion));
                            }
                            completion_ready.notify_one();
                        });
                        ++inflight;
                    }
                } catch (...) {
                    if (!first_error) first_error = std::current_exception();
                    cancelled.store(true, std::memory_order_relaxed);
                }
            };

            auto startOnePath = [&](int pid) {
                auto& state = states[pid];
                state.active = true;
                state.started_at = std::chrono::steady_clock::now();
                ++active_paths;
                peak_active_paths = std::max(peak_active_paths, active_paths);
                std::cerr << "[index-build] path " << (pid + 1) << "/" << path_count
                          << " start roles=" << plan_.paths_head_to_tail[pid].size()
                          << " unique_docs=" << state.unique_docs << "\n";
                if (state.unique_docs > 0) indexes_[pid].prepareConcurrentBuild(state.unique_docs);
            };

            auto fillActivePaths = [&] {
                std::vector<int> newly_started;
                while (!cancelled.load(std::memory_order_relaxed) &&
                       active_paths < desired_active && next_to_start < start_order.size()) {
                    const int pid = start_order[next_to_start++];
                    startOnePath(pid);
                    if (states[pid].unique_docs == 0) {
                        finishPath(pid);
                    } else {
                        newly_started.push_back(pid);
                    }
                }
                for (int pid : newly_started) scheduleNextBatch(pid);
            };

            fillActivePaths();
            while (finished_paths < path_count || inflight > 0) {
                if (first_error && inflight == 0) break;
                if (inflight == 0) {
                    fillActivePaths();
                    if (finished_paths >= path_count || first_error) break;
                }

                Completion completion;
                {
                    std::unique_lock<std::mutex> lock(completion_mutex);
                    completion_ready.wait(lock, [&] { return !completions.empty(); });
                    completion = std::move(completions.front());
                    completions.pop_front();
                }
                --inflight;
                if (completion.error && !first_error) first_error = completion.error;
                auto& state = states[completion.pid];
                --state.pending;
                if (state.pending == 0 && !cancelled.load(std::memory_order_relaxed)) {
                    try {
                        indexes_[completion.pid].commitBuildBatch(state.batch);
                        scheduleNextBatch(completion.pid);
                        fillActivePaths();
                    } catch (...) {
                        if (!first_error) first_error = std::current_exception();
                        cancelled.store(true, std::memory_order_relaxed);
                    }
                }
            }

            if (first_error) std::rethrow_exception(first_error);
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - build_t0).count();
            const double capacity = elapsed * build_threads_;
            const double busy = pool.busySeconds();
            std::cerr << "[index-build] scheduler peak_active_paths=" << peak_active_paths
                      << " peak_insertion_tasks=" << pool.peakRunning()
                      << " worker_busy_sec=" << busy
                      << " worker_idle_sec=" << std::max(0.0, capacity - busy)
                      << " worker_utilization=" << (capacity > 0.0 ? busy / capacity : 0.0) << "\n";
        }

        const uint64_t total_bytes = serializedSizeBytes();
        std::cerr << "[index-build] total size_bytes=" << total_bytes
                  << " size_mib=" << ((double)total_bytes / (1024.0 * 1024.0)) << "\n";
    }

    uint64_t serializedSizeBytes() const {
        uint64_t bytes = 8; // magic
        bytes += sizeof(n_) + sizeof(dim_) + sizeof(M_) + sizeof(ef_c_) +
                 sizeof(build_threads_) + sizeof(max_path_batch_);
        bytes += 3 * sizeof(int32_t); // role, path, and index counts
        bytes += sizeof(int64_t) + sizeof(int) * (uint64_t)role_path_.size();
        bytes += sizeof(int64_t) + sizeof(int) * (uint64_t)role_stage_.size();
        for (const auto& index : indexes_) bytes += index.serializedSizeBytes();
        return bytes;
    }

    void saveBinary(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open index cache for write: " + path);
        const char magic[8] = {'R','P','I','D','X','0','4','\0'};
        out.write(magic, sizeof(magic));
        writeScalar(out, n_);
        writeScalar(out, dim_);
        writeScalar(out, M_);
        writeScalar(out, ef_c_);
        writeScalar(out, build_threads_);
        writeScalar(out, max_path_batch_);
        int32_t role_count = (int32_t)roleCount();
        int32_t path_count = (int32_t)plan_.paths_head_to_tail.size();
        int32_t index_count = (int32_t)indexes_.size();
        writeScalar(out, role_count);
        writeScalar(out, path_count);
        writeScalar(out, index_count);
        writeVector(out, role_path_);
        writeVector(out, role_stage_);
        for (const auto& index : indexes_) index.saveBinary(out);
        if (!out) throw std::runtime_error("failed to write index cache: " + path);
    }

    void loadBinary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open index cache for read: " + path);
        char magic[8] = {};
        in.read(magic, sizeof(magic));
        const std::string got(magic, magic + sizeof(magic));
        const std::string want("RPIDX03\0", 8);
        if (got != want && got != std::string("RPIDX04\0", 8)) throw std::runtime_error("bad RolePathIndex cache magic");
        int file_n = 0, file_dim = 0, file_M = 0, file_ef_c = 0;
        int file_build_threads = 0, file_max_path_batch = 0;
        int32_t role_count = 0, path_count = 0, index_count = 0;
        readScalar(in, file_n);
        readScalar(in, file_dim);
        readScalar(in, file_M);
        readScalar(in, file_ef_c);
        readScalar(in, file_build_threads);
        readScalar(in, file_max_path_batch);
        readScalar(in, role_count);
        readScalar(in, path_count);
        readScalar(in, index_count);
        if (file_n != n_ || file_dim != dim_ || file_M != M_ || file_ef_c != ef_c_ ||
            file_build_threads != build_threads_ || file_max_path_batch != max_path_batch_) {
            throw std::runtime_error("RolePathIndex cache parameters do not match current data");
        }
        if (role_count != roleCount() || path_count != (int)plan_.paths_head_to_tail.size() || index_count != path_count) {
            throw std::runtime_error("RolePathIndex cache plan shape does not match current plan");
        }
        readVector(in, role_path_);
        readVector(in, role_stage_);
        if ((int)role_path_.size() != roleCount() || (int)role_stage_.size() != roleCount()) {
            throw std::runtime_error("RolePathIndex cache role mapping size does not match current plan");
        }
        indexes_.clear();
        indexes_.resize(index_count);
        for (int pid = 0; pid < index_count; ++pid) {
            indexes_[pid].loadBinary(in, &data_, n_, dim_, M_, ef_c_);
            indexes_[pid].finalizeLayout((int)plan_.paths_head_to_tail[pid].size());
        }
        cover_cache_.clear();
        if (!in) throw std::runtime_error("failed to read index cache: " + path);
    }

    std::vector<SearchResult> queryRole(int role, const float* q, int k, int ef_s) {
        if (role < 0 || role >= roleCount()) throw std::runtime_error("bad role id");
        int srv = servingRole(role);
        if (srv < 0 || srv >= roleCount() || role_path_[srv] < 0) return bruteForceRole(role, q, k);
        int pid = role_path_[srv], stage = role_stage_[srv];
        if (pid < 0 || pid >= (int)indexes_.size()) return bruteForceRole(role, q, k);
        if (srv == role) return indexes_[pid].search(q, k, ef_s, stage, nullptr);
        markRoleDocs(role);
        StampFilter filter(&stamp_, stamp_tag_);
        auto ans = indexes_[pid].search(q, k, ef_s, stage, &filter);
        if ((int)ans.size() < std::min(k, (int)plan_.role_docs[role].size())) return bruteForceRole(role, q, k);
        return ans;
    }

    std::vector<SearchResult> queryRoles(const std::vector<int>& roles, const float* q, int k, int ef_s,
                                         std::vector<int>* cover_out = nullptr,
                                         VirtualSearchStats* stats_out = nullptr,
                                         int edge_limit = -1,
                                         bool exact_fallback = true) {
        auto clean = cleanRoles(roles);
        auto target = unionBits(clean);
        auto cover = exactCoverRoles(clean);
        if (cover_out) *cover_out = cover;
        auto ans = virtualSearchCover(cover, target, q, k, ef_s, edge_limit, exact_fallback);
        if (stats_out) *stats_out = last_virtual_stats_;
        return ans;
    }

    std::vector<SearchResult> queryRoleApprox(int role, const float* q, int k, int ef_s) {
        if (role < 0 || role >= roleCount()) throw std::runtime_error("bad role id");
        int srv = servingRole(role);
        if (srv < 0 || srv >= roleCount() || role_path_[srv] < 0) return {};
        int pid = role_path_[srv], stage = role_stage_[srv];
        if (pid < 0 || pid >= (int)indexes_.size()) return {};
        if (srv == role) return indexes_[pid].search(q, k, ef_s, stage, nullptr);
        markRoleDocs(role);
        StampFilter filter(&stamp_, stamp_tag_);
        return indexes_[pid].search(q, k, ef_s, stage, &filter);
    }

    std::vector<SearchResult> queryRolesIndependentApprox(const std::vector<int>& roles, const float* q, int k, int ef_s,
                                                          std::vector<int>* cover_out = nullptr) {
        auto cover = exactCoverRoles(roles);
        if (cover_out) *cover_out = cover;
        std::vector<std::vector<SearchResult>> parts;
        parts.reserve(cover.size());
        for (int r : cover) parts.push_back(queryRoleApprox(r, q, k, ef_s));
        return mergeTopK(parts, k);
    }

    std::vector<SearchResult> bruteForceRole(int role, const float* q, int k) const {
        if (role < 0 || role >= roleCount()) throw std::runtime_error("bad role id");
        std::vector<SearchResult> all;
        all.reserve(plan_.role_docs[role].size());
        for (int doc : plan_.role_docs[role]) if (doc >= 0 && doc < n_) all.push_back({doc, queryLabelDist(q, doc)});
        return topKFromVector(all, k);
    }

    std::vector<SearchResult> bruteForceRolesUnion(const std::vector<int>& roles, const float* q, int k) const {
        return bruteForceTarget(unionBits(cleanRoles(roles)), q, k);
    }

    std::vector<int> exactCoverRoles(const std::vector<int>& roles) {
        auto clean = cleanRoles(roles);
        if (clean.empty()) return {};
        std::string key = coverCacheKey(clean);
        auto it = cover_cache_.find(key);
        if (it != cover_cache_.end()) return it->second;
        auto target = unionBits(clean);
        auto cover = exactCoverTarget(target, clean);
        cover_cache_[key] = cover;
        return cover;
    }

    QueryViewDiagnostics diagnoseQueryViews(const std::vector<int>& roles) {
        QueryViewDiagnostics d;
        auto clean = cleanRoles(roles);
        auto target = unionBits(clean);
        auto cover = exactCoverRoles(clean);
        std::vector<int> scan_roles;
        auto views = makeGraphViews(cover, scan_roles);
        d.assigned_roles = (int)clean.size();
        d.target_docs = popcount(target);
        d.cover_nodes = (int)cover.size();
        d.graph_views = (int)views.size();
        d.scan_roles = (int)scan_roles.size();
        d.single_graph_view = (d.graph_views == 1 && d.scan_roles == 0) ? 1 : 0;
        if (views.empty()) { d.min_view_selectivity = 0.0; return d; }
        double sum_sel = 0.0;
        d.min_view_selectivity = 1.0;
        for (const auto& v : views) {
            int sz = (v.server_role >= 0 && v.server_role < roleCount()) ? (int)plan_.role_docs[v.server_role].size() : 0;
            int inter = (v.server_role >= 0 && v.server_role < roleCount()) ? intersectCount(target, role_bits_[v.server_role]) : 0;
            d.total_view_docs += sz;
            double sel = sz > 0 ? (double)inter / (double)sz : 0.0;
            sum_sel += sel;
            d.min_view_selectivity = std::min(d.min_view_selectivity, sel);
        }
        d.avg_view_selectivity = sum_sel / std::max(1, d.graph_views);
        return d;
    }

    int roleCount() const { return (int)plan_.role_docs.size(); }
    int docCount() const { return n_; }
    int dim() const { return dim_; }
    size_t coverCacheSize() const { return cover_cache_.size(); }
    const VirtualSearchStats& lastVirtualStats() const { return last_virtual_stats_; }

private:
    struct GraphView { int cover_role = -1, server_role = -1, path_id = -1, stage = -1; };

    std::vector<float> data_;
    int n_ = 0, dim_ = 0;
    PlanData plan_;
    int M_ = 16, ef_c_ = 100;
    int build_threads_ = (int)std::max(1u, std::thread::hardware_concurrency());
    int max_path_batch_ = 8;
    std::vector<VersionedHNSW> indexes_;
    std::vector<int> role_path_, role_stage_;
    mutable std::vector<int> stamp_, seen_stamp_, visit_stamp_, neighbor_stamp_;
    mutable int stamp_tag_ = 1, seen_tag_ = 1, visit_tag_ = 1, neighbor_tag_ = 1;
    mutable std::vector<float> tmp_dist_;
    mutable VirtualSearchStats last_virtual_stats_;
    int words_ = 0;
    std::vector<std::vector<uint64_t>> role_bits_;
    std::unordered_map<std::string, std::vector<int>> cover_cache_;

    template <class T>
    static void writeScalar(std::ostream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <class T>
    static void readScalar(std::istream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    template <class T>
    static void writeVector(std::ostream& out, const std::vector<T>& values) {
        int64_t sz = (int64_t)values.size();
        writeScalar(out, sz);
        if (sz > 0) out.write(reinterpret_cast<const char*>(values.data()), sizeof(T) * values.size());
    }

    template <class T>
    static void readVector(std::istream& in, std::vector<T>& values) {
        int64_t sz = 0;
        readScalar(in, sz);
        if (sz < 0) throw std::runtime_error("bad vector size in RolePathIndex cache");
        values.resize((size_t)sz);
        if (sz > 0) in.read(reinterpret_cast<char*>(values.data()), sizeof(T) * values.size());
    }

    std::vector<int> cleanRoles(const std::vector<int>& roles) const {
        std::vector<int> clean;
        clean.reserve(roles.size());
        for (int r : roles) {
            if (r < 0 || r >= roleCount()) throw std::runtime_error("bad role id in role set");
            clean.push_back(r);
        }
        std::sort(clean.begin(), clean.end());
        clean.erase(std::unique(clean.begin(), clean.end()), clean.end());
        return clean;
    }

    void buildRoleBitsets() {
        words_ = (n_ + 63) >> 6;
        role_bits_.assign(roleCount(), std::vector<uint64_t>(words_, 0));
        for (int r = 0; r < roleCount(); ++r) {
            for (int doc : plan_.role_docs[r]) if (doc >= 0 && doc < n_) role_bits_[r][doc >> 6] |= 1ULL << (doc & 63);
        }
    }

    int servingRole(int role) const {
        int srv = (!plan_.server.empty() && role >= 0 && role < (int)plan_.server.size()) ? plan_.server[role] : role;
        return (srv < 0 || srv >= roleCount()) ? role : srv;
    }

    const float* vec(int label) const { return data_.data() + (size_t)label * dim_; }
    float l2(const float* a, const float* b) const { return simd::l2sqr(a, b, dim_); }
    float labelDist(int a, int b) const { return l2(vec(a), vec(b)); }
    float queryLabelDist(const float* q, int label) const { return l2(q, vec(label)); }

    static std::vector<SearchResult> topKFromVector(std::vector<SearchResult>& all, int k) {
        int kk = std::min(k, (int)all.size());
        if (kk <= 0) return {};
        if (kk < (int)all.size()) { std::nth_element(all.begin(), all.begin() + kk, all.end(), [](const auto& a, const auto& b){ return a.dist < b.dist; }); all.resize(kk); }
        std::sort(all.begin(), all.end(), [](const auto& a, const auto& b){ return a.dist < b.dist; });
        return all;
    }

    std::vector<SearchResult> bruteForceTarget(const std::vector<uint64_t>& target, const float* q, int k) const {
        std::vector<SearchResult> all;
        all.reserve(popcount(target));
        for (int w = 0; w < (int)target.size(); ++w) {
            uint64_t x = target[w];
            while (x) {
                int b = __builtin_ctzll(x);
                int doc = (w << 6) + b;
                if (doc < n_) all.push_back({doc, queryLabelDist(q, doc)});
                x &= x - 1;
            }
        }
        return topKFromVector(all, k);
    }

    std::vector<SearchResult> mergeTopK(const std::vector<std::vector<SearchResult>>& parts, int k) {
#ifdef ROLEWEAVER_LATENCY_PROFILE
        ProfileInterval profile_interval(merge_prune_profile.result_merge_ns);
#endif
        bumpTag(seen_stamp_, seen_tag_);
        std::vector<int> touched;
        std::vector<SearchResult> all;
        for (const auto& part : parts) for (const auto& x : part) {
            if (x.label < 0 || x.label >= n_) continue;
            if (seen_stamp_[x.label] != seen_tag_) { seen_stamp_[x.label] = seen_tag_; tmp_dist_[x.label] = x.dist; touched.push_back(x.label); }
            else if (x.dist < tmp_dist_[x.label]) tmp_dist_[x.label] = x.dist;
        }
        all.reserve(touched.size());
        for (int label : touched) all.push_back({label, tmp_dist_[label]});
        return topKFromVector(all, k);
    }

    static bool bitTest(const std::vector<uint64_t>& b, int pos) { return (b[pos >> 6] >> (pos & 63)) & 1ULL; }
    static int popcountWord(uint64_t x) { return __builtin_popcountll(x); }
    int popcount(const std::vector<uint64_t>& b) const { int s = 0; for (uint64_t w : b) s += popcountWord(w); return s; }

    std::vector<uint64_t> unionBits(const std::vector<int>& roles) const {
        std::vector<uint64_t> target(words_, 0);
        for (int r : roles) for (int w = 0; w < words_; ++w) target[w] |= role_bits_[r][w];
        return target;
    }
    bool subsetOf(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const { for (int w = 0; w < words_; ++w) if (a[w] & ~b[w]) return false; return true; }
    bool equalBits(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const { for (int w = 0; w < words_; ++w) if (a[w] != b[w]) return false; return true; }
    bool intersects(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const { for (int w = 0; w < words_; ++w) if (a[w] & b[w]) return true; return false; }
    int intersectCount(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const { int s = 0; for (int w = 0; w < words_; ++w) s += popcountWord(a[w] & b[w]); return s; }
    std::vector<uint64_t> minusBits(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const { std::vector<uint64_t> c(words_); for (int w = 0; w < words_; ++w) c[w] = a[w] & ~b[w]; return c; }
    bool emptyBits(const std::vector<uint64_t>& a) const { for (auto w : a) if (w) return false; return true; }

    void bumpTag(std::vector<int>& arr, int& tag) const { ++tag; if (tag == std::numeric_limits<int>::max()) { std::fill(arr.begin(), arr.end(), 0); tag = 1; } }
    void markRoleDocs(int role) const { bumpTag(stamp_, stamp_tag_); for (int doc : plan_.role_docs[role]) if (doc >= 0 && doc < n_) stamp_[doc] = stamp_tag_; }
    void markTargetDocs(const std::vector<uint64_t>& target) const { bumpTag(stamp_, stamp_tag_); for (int doc = 0; doc < n_; ++doc) if (bitTest(target, doc)) stamp_[doc] = stamp_tag_; }

    int roleQueryCostKey(int r) const { int srv = servingRole(r); return (int)plan_.role_docs[srv].size(); }
    bool betterCoverTie(const std::vector<int>& a, const std::vector<int>& b) const {
        if (b.empty()) return true;
        if (a.size() != b.size()) return a.size() < b.size();
        long long ca = 0, cb = 0; for (int x : a) ca += roleQueryCostKey(x); for (int x : b) cb += roleQueryCostKey(x);
        if (ca != cb) return ca < cb;
        return a < b;
    }
    std::string coverCacheKey(const std::vector<int>& roles) const { std::string s; for (int r : roles) { s += std::to_string(r); s.push_back(','); } return s; }

    std::vector<GraphView> makeGraphViews(const std::vector<int>& cover, std::vector<int>& scan_roles) const {
        std::vector<GraphView> views;
        std::unordered_set<long long> seen;
        for (int r : cover) {
            int srv = servingRole(r);
            int pid = (srv >= 0 && srv < roleCount()) ? role_path_[srv] : -1;
            int stage = (srv >= 0 && srv < roleCount()) ? role_stage_[srv] : -1;
            if (pid < 0 || pid >= (int)indexes_.size() || stage < 0) { scan_roles.push_back(r); continue; }
            long long key = (static_cast<long long>(pid) << 32) ^ static_cast<unsigned>(stage);
            if (seen.insert(key).second) views.push_back({r, srv, pid, stage});
        }
        return views;
    }

    std::vector<int> onlinePruneNeighborLabels(int current_label, const std::vector<int>& raw, int max_degree,
                                                int* prune_distance_computations = nullptr) const {
        if (max_degree <= 0 || (int)raw.size() <= max_degree) return raw;
        std::vector<std::pair<float,int>> cand;
        cand.reserve(raw.size());
        for (int nb : raw) if (nb >= 0 && nb < n_ && nb != current_label) {
            cand.push_back({labelDist(current_label, nb), nb});
            if (prune_distance_computations) ++(*prune_distance_computations);
        }
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b){ if (a.first != b.first) return a.first < b.first; return a.second < b.second; });
        std::vector<int> selected;
        selected.reserve(max_degree);
        for (auto [dq, nb] : cand) {
            bool good = true;
            for (int s : selected) {
                if (prune_distance_computations) ++(*prune_distance_computations);
                if (labelDist(nb, s) < dq) { good = false; break; }
            }
            if (good) selected.push_back(nb);
            if ((int)selected.size() >= max_degree) break;
        }
        if ((int)selected.size() < max_degree) for (auto [_, nb] : cand) {
            if (std::find(selected.begin(), selected.end(), nb) == selected.end()) selected.push_back(nb);
            if ((int)selected.size() >= max_degree) break;
        }
        return selected;
    }

    std::vector<SearchResult> virtualSearchCover(const std::vector<int>& cover, const std::vector<uint64_t>& target,
                                                 const float* q, int k, int ef_s, int edge_limit_arg, bool exact_fallback) {
        last_virtual_stats_ = VirtualSearchStats{};
        last_virtual_stats_.cover_nodes = (int)cover.size();
        last_virtual_stats_.independent_searches_avoided = std::max(0, (int)cover.size() - 1);
        int target_size = popcount(target);
        if (target_size == 0 || k <= 0) return {};
        std::vector<int> scan_roles;
        auto views = makeGraphViews(cover, scan_roles);
        last_virtual_stats_.graph_views = (int)views.size();
        if (views.empty()) { last_virtual_stats_.fallback_exact = true; return bruteForceTarget(target, q, k); }

        // Degenerate case of the virtual merged index: only one physical HNSW view is needed.
        // Use the native versioned HNSW search instead of rebuilding the layer-0 virtual edge
        // stream.  This keeps single-cover queries as fast as the ordinary one-index search,
        // while multi-cover queries still use the MSTG-style virtual edge union below.
        if (views.size() == 1 && scan_roles.empty()) {
            markTargetDocs(target);
            StampFilter single_filter(&stamp_, stamp_tag_);
            last_virtual_stats_.seed_count = 1;
            auto ans = indexes_[views[0].path_id].search(q, k, ef_s, views[0].stage, &single_filter);
            if (exact_fallback && (int)ans.size() < std::min(k, target_size)) {
                last_virtual_stats_.fallback_exact = true;
                return bruteForceTarget(target, q, k);
            }
            return ans;
        }

        markTargetDocs(target);
        StampFilter auth(&stamp_, stamp_tag_);
        using P = std::pair<float,int>;
        struct MinCmp { bool operator()(const P& a, const P& b) const { return a.first > b.first; } };
        struct MaxCmp { bool operator()(const P& a, const P& b) const { return a.first < b.first; } };
        std::priority_queue<P, std::vector<P>, MinCmp> candidate_set;
        std::priority_queue<P, std::vector<P>, MaxCmp> top_candidates;
        const int heap_cap = std::max(ef_s, k);
        bumpTag(visit_stamp_, visit_tag_);
        auto push_label = [&](int label) {
            if (label < 0 || label >= n_ || visit_stamp_[label] == visit_tag_) return;
            visit_stamp_[label] = visit_tag_;
            float d = queryLabelDist(q, label);
            ++last_virtual_stats_.distance_computations;
            if ((int)top_candidates.size() < heap_cap || d < top_candidates.top().first) {
                candidate_set.push({d, label});
                top_candidates.push({d, label});
                if ((int)top_candidates.size() > heap_cap) top_candidates.pop();
            }
        };
        for (const auto& v : views) {
            int ep = indexes_[v.path_id].greedyEntryLabel(q, v.stage);
            int raw_ep = indexes_[v.path_id].entryLabelAt(v.stage);
            if (ep >= 0) { push_label(ep); ++last_virtual_stats_.seed_count; }
            if (raw_ep >= 0 && raw_ep != ep) { push_label(raw_ep); ++last_virtual_stats_.seed_count; }
        }
        for (int r : scan_roles) for (int doc : plan_.role_docs[r]) push_label(doc);
        if (candidate_set.empty()) { last_virtual_stats_.fallback_exact = true; return bruteForceTarget(target, q, k); }

        std::vector<int> raw_edges, unique_edges;
        const int edge_limit = edge_limit_arg > 0 ? edge_limit_arg : std::max(4 * M_, 1);
        while (!candidate_set.empty()) {
            auto current = candidate_set.top(); candidate_set.pop();
            float lower_bound = top_candidates.empty() ? std::numeric_limits<float>::infinity() : top_candidates.top().first;
            if (current.first > lower_bound && (int)top_candidates.size() >= heap_cap) break;
            ++last_virtual_stats_.hops;
            int active_sources = 0;
#ifdef ROLEWEAVER_LATENCY_PROFILE
            {
            ProfileInterval profile_interval(merge_prune_profile.merge_ns);
#endif
            raw_edges.clear();
            for (const auto& v : views) {
                size_t before = raw_edges.size();
                indexes_[v.path_id].collectActiveNeighborLabels(current.second, 0, v.stage, raw_edges);
                if (raw_edges.size() > before) ++active_sources;
            }
            last_virtual_stats_.raw_edges += (int)raw_edges.size();
            bumpTag(neighbor_stamp_, neighbor_tag_);
            unique_edges.clear();
            for (int nb : raw_edges) if (nb >= 0 && nb < n_ && neighbor_stamp_[nb] != neighbor_tag_) { neighbor_stamp_[nb] = neighbor_tag_; unique_edges.push_back(nb); }
            last_virtual_stats_.unique_edges += (int)unique_edges.size();
#ifdef ROLEWEAVER_LATENCY_PROFILE
            }
#endif
            std::vector<int> pruned;
            int prune_dco = 0;
            if (active_sources > 1) {
#ifdef ROLEWEAVER_LATENCY_PROFILE
                ProfileInterval profile_interval(merge_prune_profile.prune_ns);
#endif
                // MSTG-style virtual edge merge: only when the same node exists in multiple views
                // do we rebuild the outgoing list and apply online RNG pruning.  If a node is
                // present in exactly one view, its local HNSW adjacency has already been pruned
                // during construction and should be used as-is.
                pruned = onlinePruneNeighborLabels(current.second, unique_edges, edge_limit, &prune_dco);
            } else {
                pruned = unique_edges;
            }
            last_virtual_stats_.post_prune_edges += (int)pruned.size();
            last_virtual_stats_.removed_edges += (int)unique_edges.size() - (int)pruned.size();
            last_virtual_stats_.pruned_edges += (int)pruned.size();
            last_virtual_stats_.prune_distance_computations += prune_dco;
            for (int nb : pruned) {
                if (visit_stamp_[nb] == visit_tag_) continue;
                float d = queryLabelDist(q, nb);
                ++last_virtual_stats_.distance_computations;
                visit_stamp_[nb] = visit_tag_;
                if ((int)top_candidates.size() < heap_cap || d < top_candidates.top().first) {
                    candidate_set.push({d, nb}); top_candidates.push({d, nb});
                    if ((int)top_candidates.size() > heap_cap) top_candidates.pop();
                }
            }
        }
        std::vector<SearchResult> candidates;
        candidates.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            auto x = top_candidates.top(); top_candidates.pop();
            if (auth.allowed(x.second)) candidates.push_back({x.second, x.first});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b){ if (a.dist != b.dist) return a.dist < b.dist; return a.label < b.label; });
        if ((int)candidates.size() > k) candidates.resize(k);
        if (exact_fallback && (int)candidates.size() < std::min(k, target_size)) { last_virtual_stats_.fallback_exact = true; return bruteForceTarget(target, q, k); }
        return candidates;
    }

    std::vector<int> exactCoverTarget(const std::vector<uint64_t>& target, const std::vector<int>& fallback_roles) {
        std::vector<int> cand;
        for (int r = 0; r < roleCount(); ++r) if (!plan_.role_docs[r].empty() && subsetOf(role_bits_[r], target) && intersects(role_bits_[r], target)) cand.push_back(r);
        if ((int)cand.size() <= 1024) {
            std::vector<int> keep;
            std::sort(cand.begin(), cand.end(), [&](int x, int y){ if (plan_.role_docs[x].size() != plan_.role_docs[y].size()) return plan_.role_docs[x].size() > plan_.role_docs[y].size(); return roleQueryCostKey(x) < roleQueryCostKey(y); });
            for (int r : cand) { bool dominated = false; for (int k : keep) if (subsetOf(role_bits_[r], role_bits_[k])) { dominated = true; break; } if (!dominated) keep.push_back(r); }
            cand.swap(keep);
        }
        if (cand.empty()) return fallback_roles;
        // The benchmark workload uses at most a few assigned roles per user.
        // In that common case the exact-cover problem should not enter the
        // general DFS: try all 1/2/3-role covers directly and return the best.
        std::vector<int> initial_best = fallback_roles;
        std::sort(initial_best.begin(), initial_best.end());
        initial_best.erase(std::unique(initial_best.begin(), initial_best.end()), initial_best.end());
        if ((int)initial_best.size() <= 3) {
            std::vector<int> best_small = initial_best;
            auto try_sol = [&](std::vector<int> sol) {
                std::sort(sol.begin(), sol.end());
                sol.erase(std::unique(sol.begin(), sol.end()), sol.end());
                if (betterCoverTie(sol, best_small)) best_small = sol;
            };
            for (int a : cand) if (equalBits(role_bits_[a], target)) try_sol({a});
            if ((int)best_small.size() > 1) {
                std::vector<uint64_t> u(words_);
                for (int i = 0; i < (int)cand.size(); ++i) for (int j = i + 1; j < (int)cand.size(); ++j) {
                    for (int w = 0; w < words_; ++w) u[w] = role_bits_[cand[i]][w] | role_bits_[cand[j]][w];
                    if (equalBits(u, target)) try_sol({cand[i], cand[j]});
                }
            }
            if ((int)best_small.size() > 2) {
                std::vector<uint64_t> u(words_);
                for (int i = 0; i < (int)cand.size(); ++i) for (int j = i + 1; j < (int)cand.size(); ++j) for (int k = j + 1; k < (int)cand.size(); ++k) {
                    for (int w = 0; w < words_; ++w) u[w] = role_bits_[cand[i]][w] | role_bits_[cand[j]][w] | role_bits_[cand[k]][w];
                    if (equalBits(u, target)) try_sol({cand[i], cand[j], cand[k]});
                }
            }
            return best_small;
        }
        std::vector<std::vector<int>> doc_cands(n_);
        for (int i = 0; i < (int)cand.size(); ++i) for (int doc : plan_.role_docs[cand[i]]) if (doc >= 0 && doc < n_ && bitTest(target, doc)) doc_cands[doc].push_back(i);
        std::vector<int> best = fallback_roles;
        std::sort(best.begin(), best.end()); best.erase(std::unique(best.begin(), best.end()), best.end());
        std::vector<int> cur;
        int nodes_explored = 0;
        const int NODE_LIMIT = 200000;
        auto choose_doc = [&](const std::vector<uint64_t>& uncovered)->int {
            int bd = -1, bc = std::numeric_limits<int>::max();
            for (int doc = 0; doc < n_; ++doc) if (bitTest(uncovered, doc)) {
                int cnt = 0; for (int ci : doc_cands[doc]) if (intersects(role_bits_[cand[ci]], uncovered)) ++cnt;
                if (cnt < bc) { bc = cnt; bd = doc; if (cnt <= 1) break; }
            }
            return bd;
        };
        auto dfs = [&](auto&& self, const std::vector<uint64_t>& uncovered)->void {
            if (++nodes_explored > NODE_LIMIT) return;
            if (emptyBits(uncovered)) { auto sol = cur; std::sort(sol.begin(), sol.end()); if (betterCoverTie(sol, best)) best = sol; return; }
            if ((int)cur.size() >= (int)best.size()) return;
            int remaining = popcount(uncovered), max_cover = 0;
            for (int r : cand) max_cover = std::max(max_cover, intersectCount(role_bits_[r], uncovered));
            if (max_cover <= 0) return;
            if ((int)cur.size() + (remaining + max_cover - 1) / max_cover > (int)best.size()) return;
            int doc = choose_doc(uncovered);
            if (doc < 0) return;
            std::vector<int> branch;
            for (int ci : doc_cands[doc]) if (intersects(role_bits_[cand[ci]], uncovered)) branch.push_back(cand[ci]);
            std::sort(branch.begin(), branch.end(), [&](int a, int b){ int ca = intersectCount(role_bits_[a], uncovered), cb = intersectCount(role_bits_[b], uncovered); if (ca != cb) return ca > cb; int qa = roleQueryCostKey(a), qb = roleQueryCostKey(b); if (qa != qb) return qa < qb; return a < b; });
            for (int r : branch) { cur.push_back(r); self(self, minusBits(uncovered, role_bits_[r])); cur.pop_back(); if (nodes_explored > NODE_LIMIT) return; }
        };
        dfs(dfs, target);
        return best;
    }
};

} // namespace rbac_hnsw
