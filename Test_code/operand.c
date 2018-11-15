#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#define DIGIT uint64_t

/*
 * Returns random digit
 */
inline static DIGIT rand_digit() {
    
    int i;
    DIGIT randDigit = 0;
    
    for(i = 1; i < sizeof(DIGIT) >> 1; ++i) {
        randDigit |= (DIGIT) rand() & (DIGIT)0xFFFF;
        randDigit = randDigit << 16;
    }
    randDigit |= (DIGIT) rand() & (DIGIT)0xFFFF;
    
    return randDigit;
}

/*
 * Creates a random array of digit
 */
void rand_array(DIGIT arr[], const int len) {
    int i;
    for(i = 0; i < len; ++i)  
        arr[i] = rand_digit();
}

#define N_TEST 100

/*
 * For testing purposes
 */
int main(int argc, char** argv) {
    
    int nTest, i, j;
    int len, lenRes;
    srand(time(NULL));
    
    FILE *lengths = fopen("Test_file/Lengths", "w+");
    FILE *operands = fopen("Test_file/Operands", "w");
    
    for(i = 1; i <= 100; ++i)
        fprintf(lengths, "%d\n", 10*i);
    rewind(lengths);
    
    while (fscanf(lengths, "%d", &len) == 1) {
        
        DIGIT num[len];

        fprintf(operands, "%d\n", len);
    
        for(nTest = 0; nTest < N_TEST; ++nTest) 

            for (j = 0; j < 2; ++j) {

                rand_array(num, len);   
                for (i = 0; i < len; ++i)
                    fprintf(operands, "%016" PRIX64 "    ", num[i]);
                fprintf(operands, "\n");
            }
    }
    
    return (EXIT_SUCCESS);
}
