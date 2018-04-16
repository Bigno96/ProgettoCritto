#include "A_Clmul.h"

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

// clmul fra num1 e num2, salvata in ris
__m256i clmul (__m128i val1, __m128i val2);
// clmul fra due Array, salvata in Res
void Array_Clmul(uint32_t n3, __m256i Res[], uint32_t n1, uint64_t Vett1[],  uint32_t n2, uint64_t Vett2[]);
// stampa m256i
void print__m256 (__m256i num);
// stampa m128i
void print__m128 (__m128i num);

// stampa m256i
void print__m256 (__m256i num) {
    alignas(32) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, num);
    printf("__m256 : %08X %08X %08X %08X %08X %08X %08X %08X\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

// stampa mm128i
void print__m128 (__m128i num) {
    alignas(16) uint32_t v[4];
    _mm_store_si128((__m128i*)v, num);
    printf("__m128 : %08X %08X %08X %08X\n", v[0], v[1], v[2], v[3]);
}

// clmul fra num1 e num2, salvata in ris
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

// clmul fra due Array, salvata in Res
void Array_Clmul(uint32_t n3, __m256i Res[], uint32_t n1, uint64_t Vett1[],  uint32_t n2, uint64_t Vett2[]) {

    __m256i ResTemp;
    int64_t i, j;
	__m128i V1, V2;

    for(i = 0; i <  ((n1 + 1) >> 1) + ((n2 + 1) >> 1) - 1; i++){
       Res[i] = _mm256_set_epi64x ( (uint64_t)0, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    }

    for(j = 0; j < (n2 >> 1); j++ ){        //Ciclo j n2/2 per difetto
        V2 = _mm_set_epi64x ( Vett2[2*j + 1] , Vett2[2*j]);

        for(i = 0; i < (n1 >> 1) ; i++ ){       //Ciclo i n1/2 per difetto
            V1 = _mm_set_epi64x ( Vett1[2*i + 1] , Vett1[2*i]);
            ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);     //Clmul Temporanea
            Res[j + i] ^= ResTemp;             //Xor con Res e Temp in Res
        }
    }

    if( n1 % 2 == 1){        //n1 dispari
            V1 = _mm_set_epi64x ( 0 , Vett1[n1 - 1]);

            for(j = 0; j < (n2 >> 1); j++ ){
                V2 = _mm_set_epi64x ( Vett2[2*j+1] , Vett2[2*j]);
                ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);
                Res[ j + ((n1 + 1) >> 1) - 1] ^= ResTemp;
                }
    }


    if( n2 % 2 == 1){       //n2 dispari
            V2 = _mm_set_epi64x ( 0 , Vett2[n2-1]);

            for(i = 0; i < (n1 >> 1) ; i++ ){
                V1 = _mm_set_epi64x ( Vett2[2*i+1] , Vett2[2*i]);
                ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);
                Res[ i + ((n2 + 1) >> 1) - 1] ^= ResTemp;
            }
    }

    if( (n2 % 2 == 1) && (n1 % 2 == 1) ){      //n1 e n2 dispari
            V1 = _mm_set_epi64x ( 0 , Vett1[n1-1]);
            V2 = _mm_set_epi64x ( 0 , Vett2[n2-1]);
            ResTemp = clmul( (__m128i)V1 ,(__m128i)V2);
            Res[ ((n1 + 1) >> 1) + ((n2 + 1) >> 1) -1 ] ^= ResTemp;
    }

}
