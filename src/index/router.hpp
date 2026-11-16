#pragma once
#include "role_path_index.hpp"

namespace rbac_hnsw {

enum class RouterRoute { Scan, Independent, Virtual };

struct RouterOptions {
    int tau = 4096;
    int independent_view_threshold = 3;
    double low_overlap_ratio = 1.5;
    int edge_limit = -1;
    bool exact_fallback = false;
};

inline RouterRoute selectRouterRoute(const QueryViewDiagnostics& d, const RouterOptions& options) {
    const double ratio = d.target_docs > 0 ? double(d.total_view_docs) / d.target_docs : 1e100;
    if (d.target_docs <= options.tau) return RouterRoute::Scan;
    if (d.graph_views <= options.independent_view_threshold && ratio <= options.low_overlap_ratio)
        return RouterRoute::Independent;
    return RouterRoute::Virtual;
}

// Diagnostics may be reused for repeated queries with the same authorization.
inline std::vector<SearchResult> queryRouter(RolePathIndex& index, const std::vector<int>& roles,
                                            const float* q, int k, int ef_s,
                                            const QueryViewDiagnostics& diagnostics,
                                            const RouterOptions& options = {}, RouterRoute* route_out = nullptr) {
    const auto route = selectRouterRoute(diagnostics, options);
    if (route_out) *route_out = route;
    if (route == RouterRoute::Scan) return index.bruteForceRolesUnion(roles, q, k);
    if (route == RouterRoute::Independent) return index.queryRolesIndependentApprox(roles, q, k, ef_s);
    return index.queryRoles(roles, q, k, ef_s, nullptr, nullptr, options.edge_limit, options.exact_fallback);
}

inline std::vector<SearchResult> queryRouter(RolePathIndex& index, const std::vector<int>& roles,
                                            const float* q, int k, int ef_s, const RouterOptions& options = {}) {
    return queryRouter(index, roles, q, k, ef_s, index.diagnoseQueryViews(roles), options);
}
} // namespace rbac_hnsw
