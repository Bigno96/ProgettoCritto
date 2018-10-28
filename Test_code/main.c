#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "aclmulk.h"

#define DIGIT uint64_t
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
#define N1 8
#define N2 8
#define NRES N1+N2

/*
 * For testing purposes
 */
int main(int argc, char** argv) {
    
    int i;
       
    DIGIT res_gf2x[NRES];
    DIGIT res_ack[NRES];
    DIGIT num1[N1] = {(DIGIT)0x11, (DIGIT)0x1343, (DIGIT)0x12, (DIGIT)0x9, (DIGIT)0x876, (DIGIT)0x1, (DIGIT)0xA134F, (DIGIT)0x4131A1A};
    DIGIT num2[N2] = {(DIGIT)0xAF32, (DIGIT)0xA, (DIGIT)0xAAAA, (DIGIT)0x1A2A, (DIGIT)0x2039AAF, (DIGIT)MAX32, (DIGIT)0x6AAAF, (DIGIT)0x5A12};
    
    gf2x_mul_comb(NRES, res_gf2x, N1, num1, N2, num2);
       
    ACK(NRES, res_ack, N1, num1, N2, num2);
    
    printf("\nPrint scholastic clmul:\n");
    for (i = 0; i < NRES; ++i)
        printf("0x%016" PRIX64 " ", res_gf2x[i]);
   
    printf("\nPrint ack:\n");
    for (i = 0; i < NRES; ++i)
        printf("0x%016" PRIX64 " ", res_ack[i]);
            
    return (EXIT_SUCCESS);
}
