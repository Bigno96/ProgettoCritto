#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/*
 * Read clock counter
 */
inline static uint64_t read_rdtsc() {
    uint32_t hi, lo;
    __asm__ __volatile__ ("rdtscp\n\t" : "=a"(lo), "=d"(hi)::"rcx");
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

#define N_TEST 100

/*
 * Benchmark
 */
int main(int argc, char** argv) {
    
    FILE *operands = fopen("Test_file/Operands", "r");

    uint64_t time_diff_ack[N_TEST];
    uint64_t time_diff_gf2x[N_TEST];
    uint64_t ts;
    
    int nTest, i;
    int len, lenRes;
        
    while (fscanf(operands, "%d", &len) > 0) {
        
        lenRes = len << 1;
        
        DIGIT num1[len];
        DIGIT num2[len];
        DIGIT res_gf2x[lenRes];
        DIGIT res_ack[lenRes];
        
        printf("\nlen = %d\n", len);                  // DELETE
    
        for(nTest = 0; nTest < N_TEST; ++nTest) {
            
            // copy operand from file
            i = 0;
            while (i < len && fscanf(operands, "%016" PRIX64, &num1[i++]));         
            
            i = 0;
            while (i < len && fscanf(operands, "%016" PRIX64, &num2[i++]));

            ts = read_rdtsc();                              // save clock count before gf2x multiply
            gf2x_mul_comb(lenRes, res_gf2x, len, num1, len, num2);
            time_diff_gf2x[nTest] = read_rdtsc() - ts;      // save execution time into time differencial array
            
            ts = read_rdtsc();                              // save clock count before ack multiply
            ack(lenRes, res_ack, len, num1, len, num2);
            time_diff_ack[nTest] = read_rdtsc() - ts;       // save execution time into time differencial array
        }
    }
    
    return (EXIT_SUCCESS);
}
