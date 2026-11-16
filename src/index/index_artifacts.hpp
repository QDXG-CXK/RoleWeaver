#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace roleweaver {

using Manifest = std::unordered_map<std::string, std::string>;

inline std::string artifactPath(const std::string& dir, const std::string& name) {
    return (std::filesystem::path(dir) / name).string();
}

inline void prepareOutputFile(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    if (std::filesystem::exists(path))
        std::cerr << "[output] overwriting existing file: " << path << "\n";
}

inline uint64_t hashBytes(const void* data, size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t hashFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open artifact: " + path);
    uint64_t hash = 1469598103934665603ULL;
    char buffer[65536];
    while (in) {
        in.read(buffer, sizeof(buffer));
        for (std::streamsize i = 0; i < in.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ULL;
        }
    }
    if (!in.eof()) throw std::runtime_error("failed to read artifact: " + path);
    return hash;
}

inline const std::string& required(const Manifest& manifest, const std::string& key) {
    auto it = manifest.find(key);
    if (it == manifest.end()) throw std::runtime_error("index manifest missing field: " + key);
    return it->second;
}

inline int positiveInt(const Manifest& manifest, const std::string& key) {
    const auto& text = required(manifest, key);
    size_t used = 0;
    long long value = std::stoll(text, &used);
    if (used != text.size() || value <= 0 || value > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid index manifest field: " + key);
    return static_cast<int>(value);
}

inline Manifest readManifest(const std::string& dir) {
    const auto path = artifactPath(dir, "manifest.txt");
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path + "; run roleweaver_build first");
    Manifest manifest;
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos || !manifest.emplace(line.substr(0, pos), line.substr(pos + 1)).second)
            throw std::runtime_error("invalid index manifest: " + path);
    }
    if (required(manifest, "format") != "roleweaver_index_v1")
        throw std::runtime_error("unsupported index directory; regenerate it with roleweaver_build");
    return manifest;
}

inline void validateArtifacts(const std::string& dir, const Manifest& manifest,
                              const std::vector<float>& data, int n, int dim,
                              const std::string& data_path) {
    if (required(manifest, "docs") != std::to_string(n) ||
        required(manifest, "dim") != std::to_string(dim) ||
        required(manifest, "data_size") != std::to_string(std::filesystem::file_size(data_path)) ||
        required(manifest, "data_hash") != std::to_string(hashBytes(data.data(), data.size() * sizeof(float))))
        throw std::runtime_error("base vectors do not match the built index");
    for (const std::string name : {"plan", "index"}) {
        const auto file = name == "plan" ? "plan.txt" : "index.bin";
        if (required(manifest, name + "_hash") != std::to_string(hashFile(artifactPath(dir, file))))
            throw std::runtime_error(std::string(file) + " does not match the index manifest; rebuild the index");
    }
}

} // namespace roleweaver
