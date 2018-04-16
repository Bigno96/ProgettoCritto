#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

// clmul fra num1 e num2, salvata in ris
__m256i clmul (__m128i val1, __m128i val2);
// stampa m256i
void print__m256 (__m256i num);
// stampa m128i
void print__m128 (__m128i num);

int main(int argc, char *argv[]) {
    __m256i ris;
    __m128i num1, num2;

    //inizializzo i numeri
    num1 = _mm_set_epi32(1,1,1,1);
    num2 = _mm_set_epi32(0,1,0,0);
    ris = _mm256_set_epi64x(0,0,0,0);

    ris = clmul(num1, num2);

    print__m256(ris);

    return 0;
}

__m256i clmul (__m128i val1, __m128i val2) {
    __m256i ris = _mm256_set_epi32 (-1, -1, -1, -1, -1, -1, -1, -1);
    // scheletro m256 -> c1 : c0+c1+d1+e1 . d1+c0+d0+e0 : d0
    asm ("movdqa %[val1], %%xmm0\n\t"         // val1 in xmm0
         "movdqa %[val2], %%xmm1\n\t"         // val2 in xmm1
         "movdqa %%xmm0, %%xmm2\n\t"              // val1 copiato in xmm2
         "pclmulqdq $17, %%xmm1, %%xmm2\n\t"     // a1 * b1 in xmm2, c1:c0
         "movdqa %%xmm0, %%xmm3\n\t"
         "pclmulqdq $0, %%xmm1, %%xmm3\n\t"     // a0 * b0 in xmm3, d1:d0
         "vpsrldq $8, %%xmm0, %%xmm4\n\t"       // xmm4 contiene a1 nella metà low
         "pxor %%xmm0, %%xmm4\n\t"          //a1 xor a0 in xmm4 (metà low)
         "vpsrldq $8, %%xmm1, %%xmm5\n\t"       // xmm5 contiene b1 nella metà low
         "pxor %%xmm1, %%xmm5\n\t"          //b1 xor b0 in xmm5 (metà low)
         "pclmulqdq $0, %%xmm5, %%xmm4\n\t"     // xmm4 contiene (a0 xor a1) * (b0 xor b1), e1:e0
         "pxor %%xmm3, %%xmm4\n\t"           // in xmm4 ho d1 xor e1 : d0 xor e0
         "pxor %%xmm2, %%xmm4\n\t"          // in xmm4 ho c1 xor d1 xor e1 : c0 xor d0 xor e0
         "vpsrldq $8, %%xmm4, %%xmm5\n\t"     // shifto in low di xmm5 la parte high di xmm4
         "pxor %%xmm2, %%xmm5\n\t"           // xmm5 contiene c1 : c0+c1+d1+e1
         "pslldq $8, %%xmm4\n\t"             // shifto in high di xmm4 la parte low di xmm4
         "pxor %%xmm3, %%xmm4\n\t"            // xmm4 contiene d1+c0+d0+e0 : d0
         // xmm5 : xmm4 sono la soluzione
         "vinserti128 $0, %%xmm4, %%ymm0, %%ymm0\n\t"           //copio xmm4 in low di ymm0
         "vinserti128 $1, %%xmm5, %%ymm0, %%ymm0\n\t"           //copio xmm5 in high di ymm0
         "vmovdqa %%ymm0, %[ris]\n\t"          //ret
         : [ris] "+rm" (ris)
         : [val1] "rm" (val1), [val2] "rm" (val2)
         );

    return ris;
}

// stampa m256i
void print__m256 (__m256i num) {
    alignas(32) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, num);
    printf("__m256 : %d %d %d %d %d %d %d %d\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

// stampa mm128i
void print__m128 (__m128i num) {
    alignas(16) uint32_t v[4];
    _mm_store_si128((__m128i*)v, num);
    printf("__m128 : %d %d %d %d\n", v[0], v[1], v[2], v[3]);
}


