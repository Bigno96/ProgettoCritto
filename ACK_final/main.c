#include <string.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>

#define DIGIT uint64_t

/* 
 * Prints m256i with >=c99 standard
 */
void print_m256 (const __m256i num) {
    DIGIT v[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i*)v, num);
    printf("0x%016" PRIX64 " 0x%016" PRIX64 " 0x%016" PRIX64 " 0x%016" PRIX64 "\n", v[0], v[1], v[2], v[3]);
}

/* 
 * Prints m128i with >=c99 standard
 */
void print_m128 (const __m128i num) {
    DIGIT v[2] __attribute__((aligned(32)));
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
 */
void array_clmul(const uint32_t nRes, DIGIT Res[],
                 const uint32_t n1, const DIGIT Vect1[],
                 const uint32_t n2, const DIGIT Vect2[]) {

    int i, j, k;
    bool vect1_even = true, vect2_even = true;
    DIGIT clmul_64[4] __attribute__((aligned(32)));

    __m128i V1, V2;         // 128 bits tmps to pass as input of Clmul 

    // initializing Res
    memset(Res, 0x00, nRes*sizeof(DIGIT));
     
    // setting bool for checking even
    if (n1 & 1)
        vect1_even = false;
    if (n2 & 1)
        vect2_even = false;

    for(j = n2-2; j >= 0; j -= 2) {                             // ciclying on Vect2, finishes all it if even
	V2 = _mm_set_epi64x(Vect2[j], Vect2[j+1]);		// sets first "couple" of 64's in Vect2 

        for(i = n1-2; i >= 0; i -= 2) {                         // ciclying on Vect1, finishes all it if even
            V1 = _mm_set_epi64x(Vect1[i], Vect1[i+1]);		// multiplying V2 to every blocks of Vect1 
            
            _mm256_store_si256((__m256i*)clmul_64, clmul(V1, V2));	// setting clmul result to 2 __m128i
            // saving to Res
            for (k = 3; k >= 0; k--)
                Res[j+i+k] ^= clmul_64[3-k];                   
        }
	
	// if Vect1 is odd
        if(!vect1_even) {
            V1 = _mm_set_epi64x(0, Vect1[0]);			// sets last block of Vect1 padding with zero on the upper half of V1 
	    
	    _mm256_store_si256((__m256i*)clmul_64, clmul(V1, V2));	// setting clmul result to 2 __m128i	
            // saving to Res
            for (k = 2; k >= 0; k--)
                Res[j+k] ^= clmul_64[2-k];                        
        }
    }
    
    // if Vect2 is odd
    if(!vect2_even) {
        V2 = _mm_set_epi64x(0, Vect2[0]);                   // loads last Vect2's couple, padded with zero on the upper half

        for(i = n1-2; i >= 0; i -= 2) {				// ciclying on Vect1, finishes all it if even
            V1 = _mm_set_epi64x(Vect1[i], Vect1[i+1]);		// multiplying V2 to every blocks of Vect1  
	    
	    _mm256_store_si256((__m256i*)clmul_64, clmul(V1, V2));	// setting clmul result to 2 __m128i
            // saving to Res
            for (k = 2; k >= 0; k--)
                Res[i+k] ^= clmul_64[2-k];   
        }

	// if Vect1 is odd
        if(!vect1_even) {
            V1 = _mm_set_epi64x(0, Vect1[0]);			// sets last block of Vect1 padding with zero on the upper half of V1 
	    
	    _mm256_store_si256((__m256i*)clmul_64, clmul(V1, V2));	// setting clmul result to 2 __m128i
            // saving to Res
            for (k = 1; k >= 0; k--)
                Res[k] ^= clmul_64[1-k];   
        }
    }
}

#define DIGIT_SIZE 64

/*
 * Scholastic Array Clmul bit by bit
 */
void gf2x_mul_comb(const int nr, DIGIT Res[],
                   const int na, const DIGIT A[],
                   const int nb, const DIGIT B[]) {
    
   int i, j, k;
   DIGIT u, h;

   memset(Res, 0x00, nr*sizeof(DIGIT));

   for (k = DIGIT_SIZE-1; k > 0; k--) {                     // for every bits of DIGIT
      for (i = na-1; i >= 0; i--)                           // for every element in A
         // bit masking DIGIT i of A with a 1 shifted by k position
         // if result is 1, it means there were a 1 at the k'th bit of the DIGIT    
         if (A[i] & (((DIGIT)0x1) << k))        
            for (j = nb-1; j >= 0; j--)                     // for every element in B
                Res[i+j+1] ^= B[j];                         

      // left shift all bits of the array by 1
      u = Res[na+nb-1];
      Res[na+nb-1] = u << 0x1;
      
      for (j = 1; j < na+nb; ++j) {
         h = u >> (DIGIT_SIZE-1);
         u = Res[na+nb-1-j];
         Res[na+nb-1-j] = h^(u << 0x1);
      } 
   }
   
   // takes care of last xor for bit in 0 position, which requires no shifting
   for (i = na-1; i >= 0; i--)
      if (A[i] & ((DIGIT)0x1))
         for (j = nb-1; j >= 0; j--) 
             Res[i+j+1] ^= B[j];
}

#define MAX32 UINT32_MAX
#define MAX64 UINT64_MAX
#define N1 7
#define N2 7
#define NRES N1+N2

/*
 * For testing purposes
 */
int main(int argc, char** argv) {
    
    int i;
    
    // order of memset is 3, 2, 1, 0 
    __m128i test128 = _mm_set_epi64x(0xC, 0x2);        // LB of number in position 3, HB in position 0 -> e.g., number 2003 need to be set as 3, 0, 0, 2
    __m256i test256 = _mm256_set_epi64x(MAX64, MAX64, MAX64, 0);
    
    DIGIT res_gf2x[NRES];
    DIGIT res_pcmul[NRES];
    DIGIT num1[N1] = {(DIGIT)MAX64, (DIGIT)0xC1, (DIGIT)0x176, (DIGIT)0xFCD21, (DIGIT)MAX64, (DIGIT)0x1897, (DIGIT)0x31243A};//, (DIGIT)0xFC432A};
    DIGIT num2[N2] = {(DIGIT)MAX64, (DIGIT)0x2A3F, (DIGIT)MAX64, (DIGIT)0xAA, (DIGIT)0x2FF45, (DIGIT)0x3, (DIGIT)MAX32};//, (DIGIT)0x154A};
    
    gf2x_mul_comb(NRES, res_gf2x, N1, num1, N2, num2);
    
    array_clmul(NRES, res_pcmul, N1, num1, N2, num2);
    
    printf("Print scholastic clmul:\n");
    for (i = 0; i < NRES; ++i)
        printf("0x%016" PRIX64 " ", res_gf2x[i]);
    
    printf("\nPrint pclmulqdq:\n");
    for (i = 0; i < NRES; ++i)
        printf("0x%016" PRIX64 " ", res_pcmul[i]);
            
    return (EXIT_SUCCESS);
}

