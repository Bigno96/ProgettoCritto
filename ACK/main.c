#include <string.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <inttypes.h>

#include "aclmulk.h"

#define DIGIT uint64_t

/* 
 * Prints m256i with >=c99 standard
 */
void print_m256 (const __m256i num) {
    alignas(32) DIGIT v[4];
    _mm256_store_si256((__m256i*)v, num);
    printf("0x%016" PRIX64 " 0x%016" PRIX64 " 0x%016" PRIX64 " 0x%016" PRIX64 "\n", v[0], v[1], v[2], v[3]);
}

/* 
 * Prints m128i with >=c99 standard
 */
void print_m128 (const __m128i num) {
    alignas(32) DIGIT v[2];
    _mm_store_si128((__m128i*)v, num);
    printf("0x%016" PRIX64 " 0x%016" PRIX64 "\n", v[0], v[1]);
}

/*
 *  Returns clmul between num1 and num2
 */ 
__m256i clmul (const __m128i val1, const __m128i val2) {
    __m256i ris = _mm256_setzero_si256();
    // skeleton m256 -> c1 : c0+c1+d1+e1 . d1+c0+d0+e0 : d0
    asm ("movdqu %[val1], %%xmm0\n\t"                           // val1 in xmm0
         "movdqu %[val2], %%xmm1\n\t"                           // val2 in xmm1
    
         "movdqa %%xmm0, %%xmm2\n\t"                            // val1 copied to xmm2
         "pclmulqdq $0x11, %%xmm1, %%xmm2\n\t"                  // a1 * b1 in xmm2, c1:c0   
         "movdqa %%xmm0, %%xmm3\n\t"                            // val1 copied to xmm3    
         "pclmulqdq $0x00, %%xmm1, %%xmm3\n\t"                  // a0 * b0 in xmm3, d1:d0 
    
         "vpsrldq $8, %%xmm0, %%xmm4\n\t"                       // xmm4 contains a1 in low half
         "pxor %%xmm0, %%xmm4\n\t"                              // a1 xor a0 to xmm4 (low half)
         "vpsrldq $8, %%xmm1, %%xmm5\n\t"                       // xmm5 contains b1 in low half
         "pxor %%xmm1, %%xmm5\n\t"                              // b1 xor b0 to xmm5 (low half)
    
         "pclmulqdq $0x00, %%xmm5, %%xmm4\n\t"                  // xmm4 contains (a0 xor a1) * (b0 xor b1), e1:e0
    
         "pxor %%xmm3, %%xmm4\n\t"                              // in xmm4 there's d1 xor e1 : d0 xor e0
         "pxor %%xmm2, %%xmm4\n\t"                              // in xmm4 there's c1 xor d1 xor e1 : c0 xor d0 xor e0
    
         "vpsrldq $8, %%xmm4, %%xmm5\n\t"                       // shifts to low half of xmm5 the high half of xmm4
         "pxor %%xmm2, %%xmm5\n\t"                              // xmm5 contains c1 : c0+c1+d1+e1
    
         "pslldq $8, %%xmm4\n\t"                                // shifts to high half of xmm4 the low half of xmm4
         "pxor %%xmm3, %%xmm4\n\t"                              // xmm4 contains d1+c0+d0+e0 : d0
         
        // xmm5 : xmm4 are the clmul results
         "vinserti128 $0, %%xmm4, %%ymm0, %%ymm0\n\t"           // copies xmm4 to low half of ymm0
         "vinserti128 $1, %%xmm5, %%ymm0, %%ymm0\n\t"           // copies xmm5 to high half of ymm0
         "vmovdqa %%ymm0, %[ris]\n\t"                           //ret
    
         : [ris] "=m" (ris)
         : [val1] "m" (val1), [val2] "m" (val2)
         );

    return ris;
}

/*
 *  Clmul between two Arrays, returns result in Res
 *  Undefined behavior if length of Vect1 and Vect2 are different
 */
void array_clmul(const uint32_t nRes, DIGIT Res[],
                 const uint32_t n, const DIGIT Vect1[], const DIGIT Vect2[]) {

    int i, j, k;
    __m256i clmul_res;

    __m128i V1, V2;         // 128 bits tmps to pass as input of Clmul 

    // initializing Res
    memset(Res, 0x00, nRes*sizeof(DIGIT));
     
    // vect1 and vect2 can only be both even or odd, since they have same length
    
    if (n & 1) {                    // if both are odd                                                   
        for(j = n-2; j >= 0; j -= 2) {                              // ciclying on Vect2, leaves last block
            V2 = _mm_set_epi64x(Vect2[j], Vect2[j+1]);              // sets first "couple" of 64's in Vect2 

            for(i = n-2; i >= 0; i -= 2) {                          // ciclying on Vect1, leaves last block
                V1 = _mm_set_epi64x(Vect1[i], Vect1[i+1]);          // multiplying V2 to every blocks of Vect1 

                clmul_res = clmul(V1, V2);
                Res[j+i+3] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 0);
                Res[j+i+2] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 1);
                Res[j+i+1] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 2);
                Res[j+i+0] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 3);                 
            }

            V1 = _mm_set_epi64x(0, Vect1[0]);			// sets last block of Vect1 padding with zero on the upper half of V1 

            clmul_res = clmul(V1, V2);
            Res[j+2] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 0);
            Res[j+1] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 1);
            Res[j+0] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 2);                      
        }
        
        V2 = _mm_set_epi64x(0, Vect2[0]);                       // loads last Vect2's couple, padded with zero on the upper half

        for(i = n-2; i >= 0; i -= 2) {				// ciclying on Vect1, leaves last block
            V1 = _mm_set_epi64x(Vect1[i], Vect1[i+1]);		// multiplying V2 to every blocks of Vect1  
	    
            clmul_res = clmul(V1, V2);
            Res[i+2] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 0);
            Res[i+1] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 1);
            Res[i+0] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 2); 
        }

        V1 = _mm_set_epi64x(0, Vect1[0]);			// sets last block of Vect1 padding with zero on the upper half of V1 

        clmul_res = clmul(V1, V2);
        Res[1] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 0);
        Res[0] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 1);         
    }
    else                                    // if both are even
        for(j = n-2; j >= 0; j -= 2) {                              // ciclying on Vect2
            V2 = _mm_set_epi64x(Vect2[j], Vect2[j+1]);              // sets first "couple" of 64's in Vect2 

            for(i = n-2; i >= 0; i -= 2) {                          // ciclying on Vect1
                V1 = _mm_set_epi64x(Vect1[i], Vect1[i+1]);          // multiplying V2 to every blocks of Vect1 

                clmul_res = clmul(V1, V2);
                Res[j+i+3] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 0);
                Res[j+i+2] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 1);
                Res[j+i+1] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 2);
                Res[j+i+0] ^= (DIGIT) _mm256_extract_epi64(clmul_res, 3);                 
            }
        }    
}

/*
 * Add carry less vect1 to vect2, saves result in Res
 * Behavior is undefined if Vect1 and Vect2 are not same length 
 */
void add(const uint32_t nRes, DIGIT Res[], const DIGIT Vect1[], const DIGIT Vect2[]) {     

    int i;
    for(i = 0; i < nRes; ++i)
        Res[i] = Vect1[i] ^ Vect2[i];
}

/*
 * Returns max between a and b
 */
static inline uint32_t max(const uint32_t a, const uint32_t b) {
    return (a >= b) ? a : b;
}

/*
 * Carry less multiplication between two arrays, reduction time with Karatsuba's algorithm
 * Result sets in Res
 */
void ACK(const uint32_t nRes, DIGIT Res[], 
         const uint32_t n1, const DIGIT Vect1[], 
         const uint32_t n2, const DIGIT Vect2[]) {   
    
    const uint32_t l = (max(n1, n2)+1)>>1;       // length of A0, A1, B0, B1 vectors
    const uint32_t lR = l+l;                 // length of Array Clmul's results
    
    int i, j;
    
    DIGIT A0[l], A1[l], B0[l], B1[l];  
    DIGIT tmp[lR];
    DIGIT sumA[l], sumB[l];  
    DIGIT C[lR], D[lR], E[lR];

    memset(Res, 0x00, nRes*sizeof(DIGIT));
    
    memset(tmp, 0x00, lR*sizeof(DIGIT));
    memcpy(&tmp[lR-n1], Vect1, n1*sizeof(DIGIT));   // copies vect1 to tmp, left-padded with 0
        
    memcpy(A1, tmp, l*sizeof(DIGIT));           // fills A1
    memcpy(A0, &tmp[l], l*sizeof(DIGIT));       // fills A0
    
    memset(tmp, 0x00, lR*sizeof(DIGIT));
    memcpy(&tmp[lR-n2], Vect2, n2*sizeof(DIGIT));   // copies vect2 to tmp, left-padded with 0
    
    memcpy(B1, tmp, l*sizeof(DIGIT));           // fills B1
    memcpy(B0, &tmp[l], l*sizeof(DIGIT));       // fills B0
    
    array_clmul(lR, C, l, A1, B1);
    array_clmul(lR, D, l, A0, B0);
    
    add(l, sumA, A0, A1);
    add(l, sumB, B0, B1);
    array_clmul(lR, E, l, sumA, sumB);
    
    for (i = lR-1, j = nRes-1; j >= 0 && i >= 0; i--, j--)
        Res[j] ^= D[i];
    
    for (i = lR-1, j = nRes-l-1; j >= 0 && i >= 0; i--, j--)
        Res[j] ^= (C[i]^D[i]^E[i]);
        
    for (i = lR-1, j = nRes-lR-1; j >= 0 && i >= 0; i--, j--)
        Res[j] ^= C[i];
}
