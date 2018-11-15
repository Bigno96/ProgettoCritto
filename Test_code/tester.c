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

/*
 * Update mean and sum of squares of differences from the current mean with Welford's recurrency
 */
inline static void update(double *mean, double *M2, 
                          const uint64_t value, const int n) {
    
    double delta = 0.0, delta2 = 0.0;
    
    delta = (double)value - *mean;
    *mean += delta / (double)n;
    
    delta2 = (double)value - *mean;
    *M2 += delta * delta2;
}

#define N_TEST 100

/*
 * Benchmark
 */
int main(int argc, char** argv) {
    
    FILE *operands = fopen("Test_file/Operands", "r");
    FILE *ack_result = fopen("Test_file/AckResult.dat", "w");
    FILE *gf2x_result = fopen("Test_file/Gf2xResult.dat", "w");

    uint64_t ts;
    double gf2x_mean;
    double ack_mean;
    double gf2x_M2;               // M2 is the sum of squares of differences from the current mean
    double ack_M2;                // needed for variance calc
        
    int n, i;
    int len, lenRes;
        
    while (fscanf(operands, "%d", &len) > 0) {
        
        lenRes = len << 1;
        gf2x_mean = 0.0;
        ack_mean = 0.0;
        gf2x_M2 = 0.0;               // M2 is the sum of squares of differences from the current mean
        ack_M2 = 0.0;                // needed for variance calc
        
        DIGIT num1[len];
        DIGIT num2[len];
        DIGIT res_gf2x[lenRes];
        DIGIT res_ack[lenRes];
        
        printf("\nlen = %d\n", len);                  // DELETE
          
        for(n = 1; n <= N_TEST; ++n) {
            
            // copy operand from file
            i = 0;
            while (i < len && fscanf(operands, "%016" PRIX64, &num1[i++]));         
            
            i = 0;
            while (i < len && fscanf(operands, "%016" PRIX64, &num2[i++]));

            ts = read_rdtsc();                              // save clock count before gf2x multiply
            gf2x_mul_comb(lenRes, res_gf2x, len, num1, len, num2);
            update(&gf2x_mean, &gf2x_M2, read_rdtsc()-ts, n);       // update mean and M2 with new execution time
            
            ts = read_rdtsc();                              // save clock count before ack multiply
            ack(lenRes, res_ack, len, num1, len, num2);     
            update(&ack_mean, &ack_M2, read_rdtsc()-ts, n);         // update mean and M2 with new execution time
        }
        
        fprintf(ack_result, "%d     ", len);
        fprintf(ack_result, "%.5lf    ", ack_mean);
        fprintf(ack_result, "%.5lf\n", ack_M2 / (double)(N_TEST-1));
        
        fprintf(gf2x_result, "%d     ", len);
        fprintf(gf2x_result, "%.5lf    ", gf2x_mean);
        fprintf(gf2x_result, "%.5lf\n", gf2x_M2 / (double)(N_TEST-1));
    }
    
    return (EXIT_SUCCESS);
}
