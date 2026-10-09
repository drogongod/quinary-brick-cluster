/* quinary_simd_kernel.c -- BATCHED quinary multiply-free SIMD kernel (DLL), for wiring into a real
 * training step (forward + backward-input-gradient), not just a standalone GEMV benchmark.
 * Same proven mechanism as mm_t2 (c_engine/int8_matmul.c) and rhind_simd_quinary_test.c, extended
 * to handle a batch dimension B inside one call (avoids B separate ctypes round-trips).
 *
 * Layout: S1/S2 (M,K) int8 quantized weight planes (v = s1 + 2*s2). X (K,B) int8 quantized
 * activations, COLUMN-MAJOR per batch item is NOT used here -- X is laid out (K,B) row-major
 * (K rows of B), so a given k's B values are contiguous... actually for the SIMD inner loop we
 * need 32 CONSECUTIVE k-values for one fixed b, so X must be (B,K) row-major (one row per batch
 * item, K contiguous) for this kernel -- caller transposes before calling, output is (M,B).
 *
 * BUILD: cl /O2 /openmp /arch:AVX2 /LD quinary_simd_kernel.c /Fe:quinary_simd_kernel.dll
 */
#include <stdint.h>
#include <immintrin.h>

__declspec(dllexport)
void quinary_gemm_simd(const int8_t* S1, const int8_t* S2, const int8_t* X,
                        int32_t* out, int M, int K, int B){
    const __m256i ones = _mm256_set1_epi8(1);
    int idx; int total = M * B;
    #pragma omp parallel for schedule(static)
    for(idx = 0; idx < total; idx++){
        int m = idx / B, b = idx % B;
        const int8_t* s1row = S1 + (size_t)m*K;
        const int8_t* s2row = S2 + (size_t)m*K;
        const int8_t* xrow  = X  + (size_t)b*K;    /* X is (B,K): row b, K contiguous */
        __m256i acc1 = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
        __m256i a16_1 = _mm256_setzero_si256(), a16_2 = _mm256_setzero_si256();
        int g, G = K/32, gc = 0;
        for(g=0; g<G; g++){
            __m256i xf = _mm256_loadu_si256((const __m256i*)(xrow + (size_t)g*32));
            __m256i t1 = _mm256_loadu_si256((const __m256i*)(s1row + (size_t)g*32));
            __m256i t2 = _mm256_loadu_si256((const __m256i*)(s2row + (size_t)g*32));
            __m256i p1 = _mm256_sign_epi8(xf, t1);
            __m256i p2 = _mm256_sign_epi8(xf, t2);
            a16_1 = _mm256_add_epi16(a16_1, _mm256_maddubs_epi16(ones, p1));
            a16_2 = _mm256_add_epi16(a16_2, _mm256_maddubs_epi16(ones, p2));
            if(++gc == 32){
                acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16_1)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16_1,1))));
                acc2 = _mm256_add_epi32(acc2, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16_2)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16_2,1))));
                a16_1 = _mm256_setzero_si256(); a16_2 = _mm256_setzero_si256(); gc = 0;
            }
        }
        if(gc){
            acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16_1)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16_1,1))));
            acc2 = _mm256_add_epi32(acc2, _mm256_add_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16_2)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16_2,1))));
        }
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc1), _mm256_extracti128_si256(acc1,1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E)); s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        int32_t sum1 = _mm_cvtsi128_si32(s);
        __m128i s2v = _mm_add_epi32(_mm256_castsi256_si128(acc2), _mm256_extracti128_si256(acc2,1));
        s2v = _mm_add_epi32(s2v, _mm_shuffle_epi32(s2v, 0x4E)); s2v = _mm_add_epi32(s2v, _mm_shuffle_epi32(s2v, 0xB1));
        int32_t sum2 = _mm_cvtsi128_si32(s2v);
        out[(size_t)m*B + b] = sum1 + 2*sum2;
    }
}
