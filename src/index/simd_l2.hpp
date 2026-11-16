#pragma once
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

namespace rbac_hnsw {
namespace simd {

inline float horizontal_sum_scalar(const float* x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
}

inline float l2sqr_scalar(const float* a, const float* b, int dim) {
    float s = 0.0f;
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float d0 = a[i] - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

#if defined(__AVX512F__)
inline float reduce_m512(__m512 x) {
    alignas(64) float buf[16];
    _mm512_store_ps(buf, x);
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += buf[i];
    return s;
}

inline float l2sqr_avx512(const float* a, const float* b, int dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 31 < dim; i += 32) {
        __m512 x0 = _mm512_loadu_ps(a + i);
        __m512 y0 = _mm512_loadu_ps(b + i);
        __m512 d0 = _mm512_sub_ps(x0, y0);
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(d0, d0));
        __m512 x1 = _mm512_loadu_ps(a + i + 16);
        __m512 y1 = _mm512_loadu_ps(b + i + 16);
        __m512 d1 = _mm512_sub_ps(x1, y1);
        acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(d1, d1));
    }
    for (; i + 15 < dim; i += 16) {
        __m512 x = _mm512_loadu_ps(a + i);
        __m512 y = _mm512_loadu_ps(b + i);
        __m512 d = _mm512_sub_ps(x, y);
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(d, d));
    }
    float s = reduce_m512(_mm512_add_ps(acc0, acc1));
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
#endif

#if defined(__AVX2__)
inline float reduce_m256(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline float l2sqr_avx2(const float* a, const float* b, int dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 x0 = _mm256_loadu_ps(a + i);
        __m256 y0 = _mm256_loadu_ps(b + i);
        __m256 d0 = _mm256_sub_ps(x0, y0);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
        __m256 x1 = _mm256_loadu_ps(a + i + 8);
        __m256 y1 = _mm256_loadu_ps(b + i + 8);
        __m256 d1 = _mm256_sub_ps(x1, y1);
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
    }
    for (; i + 7 < dim; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i);
        __m256 y = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(x, y);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d, d));
    }
    float s = reduce_m256(_mm256_add_ps(acc0, acc1));
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
#endif

#if defined(__SSE2__)
inline float reduce_m128(__m128 x) {
#if defined(__SSE3__)
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    return _mm_cvtss_f32(x);
#else
    alignas(16) float buf[4];
    _mm_store_ps(buf, x);
    return buf[0] + buf[1] + buf[2] + buf[3];
#endif
}
inline float l2sqr_sse(const float* a, const float* b, int dim) {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m128 x0 = _mm_loadu_ps(a + i);
        __m128 y0 = _mm_loadu_ps(b + i);
        __m128 d0 = _mm_sub_ps(x0, y0);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(d0, d0));
        __m128 x1 = _mm_loadu_ps(a + i + 4);
        __m128 y1 = _mm_loadu_ps(b + i + 4);
        __m128 d1 = _mm_sub_ps(x1, y1);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(d1, d1));
    }
    for (; i + 3 < dim; i += 4) {
        __m128 x = _mm_loadu_ps(a + i);
        __m128 y = _mm_loadu_ps(b + i);
        __m128 d = _mm_sub_ps(x, y);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(d, d));
    }
    float s = reduce_m128(_mm_add_ps(acc0, acc1));
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
#endif

inline float l2sqr(const float* a, const float* b, int dim) {
#if defined(RBAC_FORCE_SCALAR)
    return l2sqr_scalar(a, b, dim);
#elif defined(__AVX512F__)
    return l2sqr_avx512(a, b, dim);
#elif defined(__AVX2__)
    return l2sqr_avx2(a, b, dim);
#elif defined(__SSE2__)
    return l2sqr_sse(a, b, dim);
#else
    return l2sqr_scalar(a, b, dim);
#endif
}

inline const char* kernel_name() {
#if defined(RBAC_FORCE_SCALAR)
    return "scalar-forced";
#elif defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__SSE2__)
    return "sse2";
#else
    return "scalar";
#endif
}

inline void prefetch_vector(const float* p) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 1);
#else
    (void)p;
#endif
}

} // namespace simd
} // namespace rbac_hnsw
