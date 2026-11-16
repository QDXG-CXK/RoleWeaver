#include "role_path_index.hpp"
#include "index_artifacts.hpp"
#include "../optimizer/plan_optimizer.hpp"
#include <cmath>
#include <filesystem>
#include <iomanip>

using namespace rbac_hnsw;
using Clock = std::chrono::steady_clock;

struct BuildArgs {
    rbac_opt::Args optimizer;
    std::string data, index_dir;
    int ef_c = 100;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int batch = 8;
};

static BuildArgs parseArgs(int argc, char** argv) {
    BuildArgs a;
    auto value = [&](int& i) {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--base") a.data = value(i);
        else if (option == "--index-dir") a.index_dir = value(i);
        else if (option == "--roles") a.optimizer.roles_path = value(i);
        else if (option == "--M") a.optimizer.M = std::stoi(value(i));
        else if (option == "--rho") a.optimizer.rho = std::stod(value(i));
        else if (option == "--solver") a.optimizer.solver = value(i);
        else if (option == "--efc") a.ef_c = std::stoi(value(i));
        else if (option == "--build-threads") a.threads = std::stoi(value(i));
        else if (option == "--help") {
            std::cout << "Usage: roleweaver_build --base base.fbin --roles role_blocks.txt --index-dir output/run/index\n"
                         "  [--M 16 --efc 100 --build-threads N --rho 0.3\n"
                         "   --solver greedy]\n"
                         "  Existing build artifacts are overwritten with a notice. Vector count is read from --base.\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + option);
    }
    if (a.data.empty() || a.index_dir.empty() || a.optimizer.roles_path.empty())
        throw std::runtime_error("--base, --index-dir and --roles are required");
    if (a.optimizer.M <= 0 || a.ef_c <= 0 || a.threads <= 0)
        throw std::runtime_error("--M, --efc and --build-threads must be positive");
    if (!std::isfinite(a.optimizer.rho) || a.optimizer.rho < 0 || a.optimizer.rho > 1)
        throw std::runtime_error("--rho must be in [0,1]");
    if (a.optimizer.solver != "branch-cut" && a.optimizer.solver != "greedy")
        throw std::runtime_error("--solver must be branch-cut or greedy");
    return a;
}

static double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
    try {
        auto args = parseArgs(argc, argv);
        namespace fs = std::filesystem;
        fs::create_directories(args.index_dir);
        auto path = [&](const std::string& name) { return roleweaver::artifactPath(args.index_dir, name); };
        int n = 0, dim = 0;
        auto data = readFbin(args.data, n, dim);
        args.optimizer.doc_count = n;
        const auto data_hash = roleweaver::hashBytes(data.data(), data.size() * sizeof(float));
        double plan_sec = 0;
        PlanData plan;
        {
            const auto plan_start = Clock::now();
            double role_read_sec = 0;
            std::cerr << "[build] optimizing index plan\n";
            auto instance = rbac_opt::readInstance(args.optimizer, role_read_sec);
            auto result = rbac_opt::optimize(instance, args.optimizer);
            plan.doc_count = instance.doc_count;
            plan.paths_head_to_tail = result.first.paths;
            plan.server = result.first.server;
            plan_sec = secondsSince(plan_start) - role_read_sec;
            for (const auto* name : {"plan.txt", "optimizer_stats.json", "index.bin", "build_stats.json", "manifest.txt"}) {
                if (fs::exists(path(name))) {
                    std::cerr << "[build] overwriting existing build artifacts in: " << args.index_dir << "\n";
                    break;
                }
            }
            // Invalidate the previous build before replacing any of its artifacts.
            fs::remove(path("manifest.txt"));
            rbac_opt::writePlanText(path("plan.txt"), instance, result.first);
            rbac_opt::writeStats(path("optimizer_stats.json"), instance, args.optimizer, result.first, result.second);
            plan.role_docs = std::move(instance.docs);
        }
        if (plan.doc_count != n) throw std::runtime_error("plan vector count does not match base vectors");
        for (const auto& docs : plan.role_docs)
            for (int label : docs)
                if (label < 0 || label >= n) throw std::runtime_error("plan contains invalid vector label");
        std::cerr << "[build] constructing index\n";
        const auto start = Clock::now();
        RolePathIndex index(std::move(data), n, dim, std::move(plan), args.optimizer.M, args.ef_c, args.threads, args.batch);
        index.build();
        const double index_build_sec = secondsSince(start);
        index.saveBinary(path("index.bin"));
        const auto plan_hash = roleweaver::hashFile(path("plan.txt"));
        const auto index_hash = roleweaver::hashFile(path("index.bin"));
        const auto index_size = index.serializedSizeBytes();
        std::ofstream stats(path("build_stats.json"));
        stats << std::setprecision(17)
              << "{\n  \"plan_sec\": " << plan_sec << ",\n"
              << "  \"index_build_sec\": " << index_build_sec << ",\n"
              << "  \"construction_sec\": " << plan_sec + index_build_sec << "\n}\n";
        stats.close();
        if (!stats) throw std::runtime_error("failed to write build statistics");
        // Publish the manifest last: an incomplete build cannot be loaded by search.
        std::ofstream manifest(path("manifest.txt"));
        manifest << "format=roleweaver_index_v1\n"
                 << "data_size=" << fs::file_size(args.data) << "\n"
                 << "data_hash=" << data_hash << "\n"
                 << "plan_hash=" << plan_hash << "\n"
                 << "index_hash=" << index_hash << "\n"
                 << "docs=" << n << "\ndim=" << dim << "\n"
                 << "M=" << args.optimizer.M << "\nef_c=" << args.ef_c << "\n"
                 << "build_threads=" << args.threads << "\nmax_path_batch=" << args.batch << "\n";
        manifest.close();
        if (!manifest) throw std::runtime_error("failed to write index manifest");
        std::cout << std::setprecision(17) << "plan_sec," << plan_sec
                  << "\nindex_build_sec," << index_build_sec
                  << "\nconstruction_sec," << plan_sec + index_build_sec
                  << "\nindex_size_bytes," << index_size << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
