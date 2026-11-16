#pragma once
#include "simd_l2.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rbac_hnsw {

struct SearchResult {
    int label = -1;
    float dist = 0.0f;
    bool operator<(const SearchResult& o) const { return dist < o.dist; }
};

struct LabelFilter {
    virtual bool allowed(int label) const = 0;
    virtual ~LabelFilter() = default;
};

struct EdgeInterval {
    int to = -1;
    int start = 0;
    int end = -1; // -1 means infinity. Valid at version v iff start <= v && (end < 0 || v < end).
};

struct VHNode {
    int label = -1;
    int active_from = 0;
    int level = 0;
    size_t compact_layer_begin = 0;
    std::vector<std::vector<EdgeInterval>> nbrs;
};

// VERSIONED_HNSW_ENGINEERING_V6_DIAG_ENTRY_HISTORY
// Engineering changes over the previous prototype:
//   * SIMD L2 distance kernel, selected at compile time (AVX-512 / AVX2 / SSE2 / scalar).
//   * Dense label->internal table instead of unordered_map for fast global-label lookup.
//   * Timestamp visited arrays instead of unordered_set in searchLayer.
//   * No temporary active-edge vector allocation in greedy descent / search expansion.
class VersionedHNSW {
public:
    struct BuildVisitWorkspace {
        std::vector<uint32_t> visited;
        uint32_t tag = 1;
    };

    struct PreparedBuildBatch {
        std::vector<int> internals;
        int version = -1;
        int entry_snapshot = -1;
        int max_level_snapshot = -1;
        size_t consumed = 0;
    };

    VersionedHNSW() = default;
    VersionedHNSW(const std::vector<float>* data, int n, int dim, int M = 16, int ef_c = 100, uint64_t seed = 1)
        : data_(data), n_(n), dim_(dim), M_(M), ef_c_(ef_c), rng_(seed),
          label_to_internal_(std::max(0, n), -1) {
        if (!data_ || n_ <= 0 || dim_ <= 0) throw std::runtime_error("bad data in VersionedHNSW");
        maxM0_ = 2 * M_;
        level_mult_ = 1.0 / std::log((double)M_);
    }

    int size() const { return (int)nodes_.size(); }

    // Call only after all construction workers have completed. Frozen graphs are
    // immutable; compact adjacency has no version metadata for a single stage.
    void finalizeLayout(int stage_count) {
        if (frozen_) return;
        if (stage_count <= 0) throw std::runtime_error("bad stage count");
        single_stage_ = stage_count == 1;
        for (auto& node : nodes_) {
            node.compact_layer_begin = compact_offsets_.size();
            for (auto& layer : node.nbrs) {
                layer.erase(std::remove_if(layer.begin(), layer.end(), [](const auto& e) {
                    return e.start == e.end;
                }), layer.end());
                if (single_stage_) {
                    compact_offsets_.push_back(compact_neighbors_.size());
                    for (const auto& e : layer) {
                        if (e.start != 0 || e.end != -1)
                            throw std::runtime_error("single-stage graph has historical edges");
                        compact_neighbors_.push_back(e.to);
                    }
                } else {
                    layer.shrink_to_fit();
                }
            }
            if (single_stage_) std::vector<std::vector<EdgeInterval>>().swap(node.nbrs);
        }
        if (single_stage_) {
            compact_offsets_.push_back(compact_neighbors_.size());
            compact_offsets_.shrink_to_fit();
            compact_neighbors_.shrink_to_fit();
            std::vector<int>().swap(entry_by_version_);
            std::vector<int>().swap(max_level_by_version_);
        }
        nodes_.shrink_to_fit();
        frozen_ = true;
    }

    uint64_t serializedSizeBytes() const {
        uint64_t bytes = 8 + sizeof(uint8_t); // magic, layout flag
        bytes += sizeof(n_) + sizeof(dim_) + sizeof(M_) + sizeof(maxM0_) +
                 sizeof(ef_c_) + sizeof(level_mult_) + sizeof(entry_) + sizeof(max_level_);
        // Dense lookup is reconstructed from node labels at load time.
        if (!single_stage_) {
            bytes += 2 * sizeof(int64_t) + sizeof(int) *
                (uint64_t)(entry_by_version_.size() + max_level_by_version_.size());
        }
        bytes += sizeof(int32_t);
        for (const auto& node : nodes_) {
            bytes += sizeof(int) * (single_stage_ ? 2 : 3); // label, [active_from], layer count
            for (int l = 0; l <= node.level; ++l) {
                const size_t count = single_stage_
                    ? compact_offsets_[node.compact_layer_begin + l + 1] - compact_offsets_[node.compact_layer_begin + l]
                    : node.nbrs[l].size();
                bytes += sizeof(int32_t) + sizeof(int) * (single_stage_ ? 1 : 3) * (uint64_t)count;
            }
        }
        return bytes;
    }

    bool containsLabel(int label) const {
        return label >= 0 && label < (int)label_to_internal_.size() && label_to_internal_[label] >= 0;
    }

    void addPoint(int label, int version) {
        if (frozen_) throw std::runtime_error("cannot insert into frozen graph");
        if (label < 0 || label >= n_) throw std::runtime_error("label out of range");
        if (containsLabel(label)) return;
        int level = randomLevel();
        int internal = (int)nodes_.size();
        VHNode nd;
        nd.label = label;
        nd.active_from = version;
        nd.level = level;
        nd.nbrs.resize(level + 1);
        nodes_.push_back(std::move(nd));
        label_to_internal_[label] = internal;

        if (entry_ < 0) {
            entry_ = internal;
            max_level_ = level;
            recordVersionState(version);
            return;
        }

        int cur = entry_;
        float curdist = distLabelToLabel(label, nodes_[cur].label);

        // Greedy descent in upper layers.  Do not allocate active-edge buffers here.
        for (int l = max_level_; l > level; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                if (l >= (int)nodes_[cur].nbrs.size()) continue;
                const auto& es = nodes_[cur].nbrs[l];
                for (const auto& e : es) {
                    if (!edgeActive(e, version) || !nodeActive(e.to, version)) continue;
                    float d = distLabelToLabel(label, nodes_[e.to].label);
                    if (d < curdist) {
                        curdist = d;
                        cur = e.to;
                        changed = true;
                    }
                }
            }
        }

        int top_layer = std::min(level, max_level_);
        for (int l = top_layer; l >= 0; --l) {
            auto candidates = searchLayerByLabel(label, cur, ef_c_, l, version);
            auto selected = selectNeighbors(label, candidates, (l == 0 ? maxM0_ : M_));
            for (int nb : selected) addUndirectedEdge(internal, nb, version, l);
            if (!candidates.empty()) cur = candidates.front().second;
        }

        if (level > max_level_) {
            entry_ = internal;
            max_level_ = level;
        }
        recordVersionState(version);
    }

    void prepareConcurrentBuild(size_t max_nodes) {
        if (max_nodes == 0) return;
        if (!nodes_.empty()) throw std::runtime_error("concurrent build must be prepared before inserting nodes");
        nodes_.reserve(max_nodes);
        build_node_locks_.reset(new std::mutex[max_nodes]);
        build_node_lock_count_ = max_nodes;
    }

    PreparedBuildBatch prepareBuildBatch(const std::vector<int>& labels, size_t begin,
                                         size_t max_parallel, int version) {
        if (frozen_) throw std::runtime_error("cannot insert into frozen graph");
        PreparedBuildBatch batch;
        batch.version = version;
        if (begin >= labels.size() || max_parallel == 0) return batch;
        if (!build_node_locks_) throw std::runtime_error("concurrent build locks are not initialized");

        // Preserve the old construction rule: the first point is inserted serially,
        // then up to max_parallel additional points share one published snapshot.
        if (entry_ < 0) {
            addPoint(labels[begin], version);
            ++begin;
            ++batch.consumed;
        }
        const size_t end = std::min(labels.size(), begin + max_parallel);
        batch.consumed += end - begin;
        batch.internals.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            const int label = labels[i];
            if (label < 0 || label >= n_) throw std::runtime_error("label out of range");
            if (containsLabel(label)) continue;
            const int internal = (int)nodes_.size();
            if ((size_t)internal >= build_node_lock_count_) {
                throw std::runtime_error("concurrent build node capacity exceeded");
            }
            VHNode node;
            node.label = label;
            node.active_from = version;
            node.level = randomLevel();
            node.nbrs.resize(node.level + 1);
            nodes_.push_back(std::move(node));
            label_to_internal_[label] = internal;
            batch.internals.push_back(internal);
        }
        batch.entry_snapshot = entry_;
        batch.max_level_snapshot = max_level_;
        return batch;
    }

    void connectPreparedPoint(const PreparedBuildBatch& batch, size_t position,
                              BuildVisitWorkspace& workspace) {
        if (position >= batch.internals.size()) throw std::runtime_error("bad prepared build position");
        connectPreallocatedPoint(batch.internals[position], batch.version, batch.entry_snapshot,
                                 batch.max_level_snapshot, workspace);
    }

    void commitBuildBatch(const PreparedBuildBatch& batch) {
        // Publish a deterministic entry point only after every insertion in the batch is complete.
        for (int internal : batch.internals) {
            if (nodes_[internal].level > max_level_) {
                entry_ = internal;
                max_level_ = nodes_[internal].level;
            }
        }
        if (!batch.internals.empty()) recordVersionState(batch.version);
    }

    void finishConcurrentBuild() {
        build_node_locks_.reset();
        build_node_lock_count_ = 0;
    }

    std::vector<SearchResult> search(const float* query, int k, int ef_s, int version, const LabelFilter* filter = nullptr) const {
        if (entry_ < 0 || k <= 0) return {};
        int cur = findBestActiveEntry(version);
        if (cur < 0) return {};
        int search_max_level = maxLevelAt(version);
        float curdist = distQueryToLabel(query, nodes_[cur].label);
        for (int l = search_max_level; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                forEachNeighbor(cur, l, version, [&](int nb) {
                    int nb_label = nodes_[nb].label;
                    simd::prefetch_vector(vec(nb_label));
                    float d = distQueryToLabel(query, nb_label);
                    if (d < curdist) {
                        curdist = d;
                        cur = nb;
                        changed = true;
                    }
                });
            }
        }
        auto cand = searchLayerByVector(query, cur, std::max(ef_s, k), 0, version);
        std::vector<SearchResult> out;
        out.reserve(k);
        for (auto& p : cand) {
            int label = nodes_[p.second].label;
            if (filter && !filter->allowed(label)) continue;
            out.push_back({label, p.first});
            if ((int)out.size() >= k) break;
        }
        return out;
    }

    bool labelActiveAt(int label, int version) const {
        int internal = labelToInternal(label);
        return internal >= 0 && nodeActive(internal, version);
    }

    int entryLabelAt(int version) const {
        int internal = findBestActiveEntry(version);
        return internal < 0 ? -1 : nodes_[internal].label;
    }

    int entryLabelForQuery(const float* query, int version) const { return greedyEntryLabel(query, version); }

    int greedyEntryLabel(const float* query, int version) const {
        int cur = findBestActiveEntry(version);
        if (cur < 0) return -1;
        int search_max_level = maxLevelAt(version);
        float curdist = distQueryToLabel(query, nodes_[cur].label);
        for (int l = search_max_level; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                forEachNeighbor(cur, l, version, [&](int nb) {
                    int nb_label = nodes_[nb].label;
                    float d = distQueryToLabel(query, nb_label);
                    if (d < curdist) {
                        curdist = d;
                        cur = nb;
                        changed = true;
                    }
                });
            }
        }
        return nodes_[cur].label;
    }

    void collectActiveNeighborLabels(int label, int layer, int version, std::vector<int>& out) const {
        int internal = labelToInternal(label);
        if (internal < 0 || !nodeActive(internal, version)) return;
        forEachNeighbor(internal, layer, version, [&](int nb) {
            out.push_back(nodes_[nb].label);
        });
    }

    const char* distanceKernelName() const { return simd::kernel_name(); }

    void saveBinary(std::ostream& out) const {
        const char magic[8] = {'V','H','N','S','W','0','2','\0'};
        out.write(magic, sizeof(magic));
        writeScalar(out, uint8_t(single_stage_));
        writeScalar(out, n_);
        writeScalar(out, dim_);
        writeScalar(out, M_);
        writeScalar(out, maxM0_);
        writeScalar(out, ef_c_);
        writeScalar(out, level_mult_);
        writeScalar(out, entry_);
        writeScalar(out, max_level_);
        if (!single_stage_) {
            writeVector(out, entry_by_version_);
            writeVector(out, max_level_by_version_);
        }
        writeScalar(out, int32_t(nodes_.size()));
        for (const auto& node : nodes_) {
            writeScalar(out, node.label);
            if (!single_stage_) writeScalar(out, node.active_from);
            writeScalar(out, int32_t(node.level + 1));
            for (int l = 0; l <= node.level; ++l) {
                if (single_stage_) {
                    size_t begin = compact_offsets_[node.compact_layer_begin + l];
                    size_t end = compact_offsets_[node.compact_layer_begin + l + 1];
                    writeScalar(out, int32_t(end - begin));
                    if (end > begin) out.write(reinterpret_cast<const char*>(compact_neighbors_.data() + begin),
                                               sizeof(int) * (end - begin));
                } else {
                    const auto& layer = node.nbrs[l];
                    writeScalar(out, int32_t(layer.size()));
                    for (const auto& edge : layer) {
                        writeScalar(out, edge.to);
                        writeScalar(out, edge.start);
                        writeScalar(out, edge.end);
                    }
                }
            }
        }
        if (!out) throw std::runtime_error("failed to write VersionedHNSW index");
    }

    void loadBinary(std::istream& in, const std::vector<float>* data, int n, int dim, int M, int ef_c) {
        char magic[8] = {};
        in.read(magic, sizeof(magic));
        const std::string got(magic, sizeof(magic));
        const bool legacy = got == std::string("VHNSW01\0", 8);
        if (!legacy && got != std::string("VHNSW02\0", 8))
            throw std::runtime_error("bad VersionedHNSW index magic");
        single_stage_ = false;
        frozen_ = false;
        compact_offsets_.clear();
        compact_neighbors_.clear();
        entry_by_version_.clear();
        max_level_by_version_.clear();
        if (!legacy) {
            uint8_t flag = 0;
            readScalar(in, flag);
            if (flag > 1) throw std::runtime_error("bad VersionedHNSW layout flag");
            single_stage_ = flag != 0;
        }
        int file_n = 0, file_dim = 0, file_M = 0, file_ef_c = 0;
        readScalar(in, file_n);
        readScalar(in, file_dim);
        readScalar(in, file_M);
        readScalar(in, maxM0_);
        readScalar(in, file_ef_c);
        readScalar(in, level_mult_);
        readScalar(in, entry_);
        readScalar(in, max_level_);
        if (file_n != n || file_dim != dim || file_M != M || file_ef_c != ef_c)
            throw std::runtime_error("VersionedHNSW index parameters do not match current data");
        data_ = data;
        n_ = n;
        dim_ = dim;
        M_ = M;
        ef_c_ = ef_c;
        if (legacy) readVector(in, label_to_internal_);
        else label_to_internal_.assign(n_, -1);
        if (!single_stage_) {
            readVector(in, entry_by_version_);
            readVector(in, max_level_by_version_);
        }
        int32_t node_count = 0;
        readScalar(in, node_count);
        if (node_count < 0 || node_count > n_) throw std::runtime_error("bad VersionedHNSW node count");
        nodes_.assign(node_count, {});
        for (int i = 0; i < node_count; ++i) {
            auto& node = nodes_[i];
            readScalar(in, node.label);
            if (node.label < 0 || node.label >= n_) throw std::runtime_error("bad VersionedHNSW label");
            if (!legacy) {
                if (label_to_internal_[node.label] >= 0) throw std::runtime_error("duplicate VersionedHNSW label");
                label_to_internal_[node.label] = i;
            }
            if (!single_stage_) readScalar(in, node.active_from);
            if (legacy) readScalar(in, node.level);
            int32_t layer_count = 0;
            readScalar(in, layer_count);
            if (layer_count <= 0) throw std::runtime_error("bad VersionedHNSW layer count");
            node.level = layer_count - 1;
            node.compact_layer_begin = compact_offsets_.size();
            if (!single_stage_) node.nbrs.resize(layer_count);
            for (int l = 0; l < layer_count; ++l) {
                int32_t edge_count = 0;
                readScalar(in, edge_count);
                if (edge_count < 0) throw std::runtime_error("bad VersionedHNSW edge count");
                if (single_stage_) {
                    compact_offsets_.push_back(compact_neighbors_.size());
                    for (int e = 0; e < edge_count; ++e) {
                        int to = -1;
                        readScalar(in, to);
                        if (to < 0 || to >= node_count) throw std::runtime_error("bad VersionedHNSW neighbor");
                        compact_neighbors_.push_back(to);
                    }
                } else {
                    auto& layer = node.nbrs[l];
                    layer.resize(edge_count);
                    for (auto& edge : layer) {
                        readScalar(in, edge.to);
                        readScalar(in, edge.start);
                        readScalar(in, edge.end);
                        if (edge.to < 0 || edge.to >= node_count) throw std::runtime_error("bad VersionedHNSW neighbor");
                    }
                }
            }
        }
        if (single_stage_) {
            compact_offsets_.push_back(compact_neighbors_.size());
            compact_offsets_.shrink_to_fit();
            compact_neighbors_.shrink_to_fit();
        }
        frozen_ = !legacy;
        visited_.clear();
        visit_tag_ = 1;
        if (!in) throw std::runtime_error("failed to read VersionedHNSW index");
    }

private:
    const std::vector<float>* data_ = nullptr;
    int n_ = 0;
    int dim_ = 0;
    int M_ = 16;
    int maxM0_ = 32;
    int ef_c_ = 100;
    double level_mult_ = 1.0;
    mutable std::mt19937_64 rng_{1};
    std::vector<VHNode> nodes_;
    bool single_stage_ = false;
    bool frozen_ = false;
    std::vector<size_t> compact_offsets_;
    std::vector<int> compact_neighbors_;
    std::vector<int> label_to_internal_;
    int entry_ = -1;
    int max_level_ = -1;
    // Versioned views must use the entry point and maximum layer that were valid
    // when that logical role stage was finished.  Using the final head entry for
    // an older stage can silently fall back to an arbitrary active node and hurts
    // extracted-view recall.  These arrays record the construction state after
    // each version/stage.
    std::vector<int> entry_by_version_;
    std::vector<int> max_level_by_version_;

    mutable std::vector<uint32_t> visited_;
    mutable uint32_t visit_tag_ = 1;
    std::unique_ptr<std::mutex[]> build_node_locks_;
    size_t build_node_lock_count_ = 0;

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
        if (sz < 0) throw std::runtime_error("bad vector size in VersionedHNSW index");
        values.resize((size_t)sz);
        if (sz > 0) in.read(reinterpret_cast<char*>(values.data()), sizeof(T) * values.size());
    }

    int randomLevel() {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        double r = std::max(dis(rng_), 1e-12);
        return (int)(-std::log(r) * level_mult_);
    }

    const float* vec(int label) const { return data_->data() + (size_t)label * dim_; }

    float distQueryToLabel(const float* q, int label) const {
        return simd::l2sqr(q, vec(label), dim_);
    }

    float distLabelToLabel(int a, int b) const {
        return simd::l2sqr(vec(a), vec(b), dim_);
    }

    int labelToInternal(int label) const {
        return (label >= 0 && label < (int)label_to_internal_.size()) ? label_to_internal_[label] : -1;
    }

    bool edgeActive(const EdgeInterval& e, int version) const {
        return e.start <= version && (e.end < 0 || version < e.end);
    }

    bool nodeActive(int internal, int version) const {
        return internal >= 0 && internal < (int)nodes_.size() && (single_stage_ ? version >= 0 : nodes_[internal].active_from <= version);
    }

    template<class F>
    void forEachNeighbor(int internal, int layer, int version, F&& visit) const {
        const auto& node = nodes_[internal];
        if (layer < 0 || layer > node.level) return;
        if (single_stage_) {
            if (version < 0) return;
            const size_t offset = node.compact_layer_begin + layer;
            for (size_t i = compact_offsets_[offset]; i < compact_offsets_[offset + 1]; ++i)
                visit(compact_neighbors_[i]);
        } else {
            for (const auto& e : node.nbrs[layer])
                if (edgeActive(e, version) && nodeActive(e.to, version)) visit(e.to);
        }
    }

    int findAnyActive(int version) const {
        for (int i = 0; i < (int)nodes_.size(); ++i) if (nodeActive(i, version)) return i;
        return -1;
    }

    void recordVersionState(int version) {
        if (version < 0) return;
        if ((int)entry_by_version_.size() <= version) {
            entry_by_version_.resize(version + 1, -1);
            max_level_by_version_.resize(version + 1, -1);
        }
        entry_by_version_[version] = entry_;
        max_level_by_version_[version] = max_level_;
    }

    int findBestActiveEntry(int version) const {
        if (single_stage_) return version >= 0 ? entry_ : -1;
        if (version >= 0 && version < (int)entry_by_version_.size()) {
            int e = entry_by_version_[version];
            if (e >= 0 && nodeActive(e, version)) return e;
        }
        for (int v = std::min(version, (int)entry_by_version_.size() - 1); v >= 0; --v) {
            int e = entry_by_version_[v];
            if (e >= 0 && nodeActive(e, version)) return e;
        }
        if (entry_ >= 0 && nodeActive(entry_, version)) return entry_;
        return findAnyActive(version);
    }

    int maxLevelAt(int version) const {
        if (single_stage_) return max_level_;
        if (version >= 0 && version < (int)max_level_by_version_.size() && max_level_by_version_[version] >= 0)
            return max_level_by_version_[version];
        for (int v = std::min(version, (int)max_level_by_version_.size() - 1); v >= 0; --v)
            if (max_level_by_version_[v] >= 0) return max_level_by_version_[v];
        return max_level_;
    }

    void resetVisited() const {
        if (visited_.size() < nodes_.size()) visited_.resize(nodes_.size(), 0);
        ++visit_tag_;
        if (visit_tag_ == std::numeric_limits<uint32_t>::max()) {
            std::fill(visited_.begin(), visited_.end(), 0);
            visit_tag_ = 1;
        }
    }

    bool isVisited(int internal) const { return visited_[internal] == visit_tag_; }
    void setVisited(int internal) const { visited_[internal] = visit_tag_; }

    void resetBuildVisited(BuildVisitWorkspace& workspace) const {
        if (workspace.visited.size() < nodes_.size()) workspace.visited.resize(nodes_.size(), 0);
        ++workspace.tag;
        if (workspace.tag == std::numeric_limits<uint32_t>::max()) {
            std::fill(workspace.visited.begin(), workspace.visited.end(), 0);
            workspace.tag = 1;
        }
    }

    std::vector<int> activeNeighborSnapshot(int internal, int layer, int version) const {
        std::vector<int> neighbors;
        if (internal < 0 || internal >= (int)nodes_.size()) return neighbors;
        std::lock_guard<std::mutex> guard(build_node_locks_[internal]);
        if (layer < 0 || layer >= (int)nodes_[internal].nbrs.size()) return neighbors;
        for (const auto& edge : nodes_[internal].nbrs[layer]) {
            if (edgeActive(edge, version) && nodeActive(edge.to, version)) neighbors.push_back(edge.to);
        }
        return neighbors;
    }

    std::vector<std::pair<float,int>> searchLayerByLabelConcurrent(
            int query_label, int entry, int ef_width, int layer, int version,
            BuildVisitWorkspace& workspace) const {
        using P = std::pair<float,int>;
        struct MinCmp { bool operator()(const P& a, const P& b) const { return a.first > b.first; } };
        struct MaxCmp { bool operator()(const P& a, const P& b) const { return a.first < b.first; } };
        std::priority_queue<P, std::vector<P>, MinCmp> candidates;
        std::priority_queue<P, std::vector<P>, MaxCmp> top;
        if (entry < 0 || entry >= (int)nodes_.size()) return {};
        resetBuildVisited(workspace);
        const float initial_dist = distLabelToLabel(query_label, nodes_[entry].label);
        candidates.push({initial_dist, entry});
        top.push({initial_dist, entry});
        workspace.visited[entry] = workspace.tag;
        while (!candidates.empty()) {
            const auto current = candidates.top();
            candidates.pop();
            const float worst = top.empty() ? std::numeric_limits<float>::infinity() : top.top().first;
            if (current.first > worst && (int)top.size() >= ef_width) break;
            for (int neighbor : activeNeighborSnapshot(current.second, layer, version)) {
                if (workspace.visited[neighbor] == workspace.tag) continue;
                workspace.visited[neighbor] = workspace.tag;
                const float distance = distLabelToLabel(query_label, nodes_[neighbor].label);
                if ((int)top.size() < ef_width || distance < top.top().first) {
                    candidates.push({distance, neighbor});
                    top.push({distance, neighbor});
                    if ((int)top.size() > ef_width) top.pop();
                }
            }
        }
        std::vector<P> result;
        result.reserve(top.size());
        while (!top.empty()) {
            result.push_back(top.top());
            top.pop();
        }
        std::sort(result.begin(), result.end(), [](const P& a, const P& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        return result;
    }

    void addUndirectedEdgeConcurrent(int a, int b, int version, int layer) {
        if (a == b) return;
        const int first = std::min(a, b);
        const int second = std::max(a, b);
        std::unique_lock<std::mutex> first_lock(build_node_locks_[first]);
        std::unique_lock<std::mutex> second_lock(build_node_locks_[second]);
        ensureLayer(a, layer);
        ensureLayer(b, layer);
        addDirectedEdge(a, b, version, layer);
        addDirectedEdge(b, a, version, layer);
        pruneNode(a, version, layer, layer == 0 ? maxM0_ : M_);
        pruneNode(b, version, layer, layer == 0 ? maxM0_ : M_);
    }

    void connectPreallocatedPoint(int internal, int version, int entry_snapshot,
                                  int max_level_snapshot, BuildVisitWorkspace& workspace) {
        if (entry_snapshot < 0) return;
        const int label = nodes_[internal].label;
        const int level = nodes_[internal].level;
        int current = entry_snapshot;
        float current_dist = distLabelToLabel(label, nodes_[current].label);
        for (int layer = max_level_snapshot; layer > level; --layer) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int neighbor : activeNeighborSnapshot(current, layer, version)) {
                    const float distance = distLabelToLabel(label, nodes_[neighbor].label);
                    if (distance < current_dist) {
                        current_dist = distance;
                        current = neighbor;
                        changed = true;
                    }
                }
            }
        }
        const int top_layer = std::min(level, max_level_snapshot);
        for (int layer = top_layer; layer >= 0; --layer) {
            auto candidates = searchLayerByLabelConcurrent(label, current, ef_c_,
                                                           layer, version, workspace);
            auto selected = selectNeighbors(label, candidates, layer == 0 ? maxM0_ : M_);
            for (int neighbor : selected) addUndirectedEdgeConcurrent(internal, neighbor, version, layer);
            if (!candidates.empty()) current = candidates.front().second;
        }
    }

    std::vector<std::pair<float,int>> searchLayerByLabel(int query_label, int entry, int ef_width, int layer, int version) const {
        const float* q = vec(query_label);
        return searchLayerByVector(q, entry, ef_width, layer, version);
    }

    std::vector<std::pair<float,int>> searchLayerByVector(const float* query, int entry, int ef_width, int layer, int version) const {
        using P = std::pair<float,int>;
        struct MinCmp { bool operator()(const P& a, const P& b) const { return a.first > b.first; } };
        struct MaxCmp { bool operator()(const P& a, const P& b) const { return a.first < b.first; } };
        std::priority_queue<P, std::vector<P>, MinCmp> cand;
        std::priority_queue<P, std::vector<P>, MaxCmp> top;
        if (entry < 0 || entry >= (int)nodes_.size()) return {};
        resetVisited();
        float d0 = distQueryToLabel(query, nodes_[entry].label);
        cand.push({d0, entry});
        top.push({d0, entry});
        setVisited(entry);
        while (!cand.empty()) {
            auto cur = cand.top(); cand.pop();
            float worst = top.empty() ? std::numeric_limits<float>::infinity() : top.top().first;
            if (cur.first > worst && (int)top.size() >= ef_width) break;
            forEachNeighbor(cur.second, layer, version, [&](int nb) {
                if (isVisited(nb)) return;
                setVisited(nb);
                int nb_label = nodes_[nb].label;
                simd::prefetch_vector(vec(nb_label));
                float d = distQueryToLabel(query, nb_label);
                if ((int)top.size() < ef_width || d < top.top().first) {
                    cand.push({d, nb});
                    top.push({d, nb});
                    if ((int)top.size() > ef_width) top.pop();
                }
            });
        }
        std::vector<P> out;
        out.reserve(top.size());
        while (!top.empty()) { out.push_back(top.top()); top.pop(); }
        std::sort(out.begin(), out.end(), [](const P& a, const P& b){
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        return out;
    }

    std::vector<int> selectNeighbors(int query_label, const std::vector<std::pair<float,int>>& candidates, int max_degree) const {
        std::vector<int> selected;
        selected.reserve(max_degree);
        for (auto [dq, cand] : candidates) {
            if (nodes_[cand].label == query_label) continue;
            bool good = true;
            for (int s : selected) {
                float ds = distLabelToLabel(nodes_[cand].label, nodes_[s].label);
                if (ds < dq) { good = false; break; }
            }
            if (good) selected.push_back(cand);
            if ((int)selected.size() >= max_degree) break;
        }
        if ((int)selected.size() < max_degree) {
            for (auto [_, cand] : candidates) {
                if (nodes_[cand].label == query_label) continue;
                if (std::find(selected.begin(), selected.end(), cand) == selected.end()) selected.push_back(cand);
                if ((int)selected.size() >= max_degree) break;
            }
        }
        return selected;
    }

    void addUndirectedEdge(int a, int b, int version, int layer) {
        if (a == b) return;
        ensureLayer(a, layer);
        ensureLayer(b, layer);
        addDirectedEdge(a, b, version, layer);
        addDirectedEdge(b, a, version, layer);
        pruneNode(a, version, layer, layer == 0 ? maxM0_ : M_);
        pruneNode(b, version, layer, layer == 0 ? maxM0_ : M_);
    }

    void ensureLayer(int internal, int layer) {
        if ((int)nodes_[internal].nbrs.size() <= layer) nodes_[internal].nbrs.resize(layer + 1);
    }

    void addDirectedEdge(int from, int to, int version, int layer) {
        auto& es = nodes_[from].nbrs[layer];
        for (auto& e : es) if (e.to == to && edgeActive(e, version)) return;
        es.push_back({to, version, -1});
    }

    void pruneNode(int internal, int version, int layer, int max_degree) {
        if (internal < 0 || internal >= (int)nodes_.size() || layer >= (int)nodes_[internal].nbrs.size()) return;
        const auto& es = nodes_[internal].nbrs[layer];
        int active_count = 0;
        for (const auto& e : es) if (edgeActive(e, version)) ++active_count;
        if (active_count <= max_degree) return;
        std::vector<std::pair<float,int>> candidates;
        candidates.reserve(active_count);
        int qlabel = nodes_[internal].label;
        for (const auto& e : es) if (edgeActive(e, version) && nodeActive(e.to, version)) {
            candidates.push_back({distLabelToLabel(qlabel, nodes_[e.to].label), e.to});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b){
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        auto keep = selectNeighbors(qlabel, candidates, max_degree);
        std::vector<int> keep_mark;
        keep_mark.reserve(keep.size());
        for (int x : keep) keep_mark.push_back(x);
        for (auto& e : nodes_[internal].nbrs[layer]) {
            if (!edgeActive(e, version)) continue;
            if (std::find(keep_mark.begin(), keep_mark.end(), e.to) == keep_mark.end()) e.end = version;
        }
    }
};

} // namespace rbac_hnsw
