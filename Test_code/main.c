#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#include "ack.h"

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
#define N1 127
#define N2 127
#define NRES N1 + N2
#define N_TEST 100
#define MN_DIGIT 0x7F          
/*
 * For testing purposes
 */
int main(int argc, char** argv) {
    
    DIGIT num1[N1];
    DIGIT num2[N2];
    DIGIT res_gf2x[NRES];
    DIGIT res_ack[NRES];

    int nTest, i, j;
    DIGIT randDigit;
    int randN1, randN2, randNRes;
    int error_bool = 0;
    srand(time(NULL));
    
    for(nTest = 0; nTest < N_TEST; ++nTest) {
        
        randN1 = rand() & MN_DIGIT;
        while (!randN1) 
            randN1 = rand() & MN_DIGIT;
        
        randN2 = rand() & MN_DIGIT;
        while (!randN2)
            randN2 = rand() & MN_DIGIT;
        
        randNRes = randN1 + randN2;
        
        for(i = 0; i < randN1; ++i) {
            randDigit = 0;
            for(j = 1; j < sizeof(DIGIT) >> 1; ++j) {
                randDigit |= (DIGIT) rand() & (DIGIT)0xFFFF;
                randDigit = randDigit << 16;
            }
            randDigit |= (DIGIT) rand() & (DIGIT)0xFFFF;
            num1[i] = randDigit;
        }

        for(i = 0; i < randN2; ++i) {
            randDigit = 0;
            for(j = 1; j < sizeof(DIGIT) >> 1; ++j) {
                randDigit |= (DIGIT) rand() & (DIGIT)0xFFFF;
                randDigit = randDigit << 16;
            }
            randDigit |= (DIGIT) rand() & (DIGIT)0xFFFF;
            num2[i] = randDigit;
        }
        
        gf2x_mul_comb(randNRes, res_gf2x, randN1, num1, randN2, num2);
        
        ACK(randNRes, res_ack, randN1, num1, randN2, num2);

        for (i = 0; i < randNRes; ++i) 
            if((res_gf2x[i] ^ res_ack[i])) {
                
                printf("\nConfrontation error\n");
                
                printf("\nPrint scholastic clmul:\n");
                for (i = 0; i < randNRes; ++i)
                    printf("0x%016" PRIX64 " ", res_gf2x[i]);
               
                printf("\nPrint ack:\n");
                for (i = 0; i < randNRes; ++i)
                    printf("0x%016" PRIX64 " ", res_ack[i]);    
                
                error_bool = 1;
            }
            
        if (!error_bool)
            printf("\nConfrontation success");  
        error_bool = 0;
    }
    
    return (EXIT_SUCCESS);
}
