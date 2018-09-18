#include <string.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <inttypes.h>

#define DIGIT uint64_t
#define DIGIT_SIZE 64
#define MAX_SIZE 100

void print__m128 (__m128i num);
void print__m256 (__m256i num);
__m256i clmul (__m128i val1, __m128i val2);
void Array_Clmul(uint32_t n3, DIGIT Res[], uint32_t n1, DIGIT Vett1[],  uint32_t n2, DIGIT Vett2[]);
void Add(uint32_t n3, DIGIT Res[], uint32_t n1, DIGIT Vett1[], uint32_t n2, DIGIT Vett2[]);
void ACK(uint32_t n3, DIGIT Res[], uint32_t n1, DIGIT Vett1[], uint32_t n2, DIGIT Vett2[]);

int main(int argc, char *argv[])
{
    uint32_t n1 = 8, n2 = 8;
    uint32_t n3 = n1+n2;
    uint32_t i = 0;
    DIGIT Vett1[n1], Vett2[n2], Res[n3];

    Vett1[0] = 0X00;
    Vett1[1] = 0x00;
    Vett1[2] = UINT64_MAX;
    Vett1[3] = 0x00;
    Vett1[4] = 0x00;
    Vett1[5] = 0x00;
    Vett1[6] = 0x00;
    Vett1[7] = 0X00;

    Vett2[0] = 0X00;
    Vett2[1] = 0x00;
    Vett2[2] = 0x00;
    Vett2[3] = 0X00;
    Vett2[4] = 0x00;
    Vett2[5] = 0x00;
    Vett2[6] = UINT64_MAX;
    Vett2[7] = 0x00;

    ACK(n3, Res, n1, Vett1, n2, Vett2);

    for (i=0; i<n3-1; i+=2)
        printf("\nres[%d] res[%d]: %016lX %016lX\n",i, i+1, Res[i], Res[i+1]);

    return 0;
}

// stampa m256i
void print__m256 (__m256i num) {
    alignas(32) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, num);
    printf("0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n", v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
}

// stampa mm128i
void print__m128 (__m128i num) {
    alignas(16) uint32_t v[4];
    _mm_store_si128((__m128i*)v, num);
    printf("0x%08X 0x%08X 0x%08X 0x%08X\n", v[3], v[2], v[1], v[0]);
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
         : [ris] "=m" (ris)
         : [val1] "m" (val1), [val2] "m" (val2)
         );

    return ris;
}

// clmul fra due Array, salvata in Res
void Array_Clmul(uint32_t n3, DIGIT Res[], uint32_t n1, DIGIT Vett1[],  uint32_t n2, DIGIT Vett2[]) {

    int64_t i=0, j=0;

    __m128i V1, V2;                                     //128 bit da passare in ingresso a clmul

    //inizializzo ResTemp
    memset(Res, 0x00, n3*sizeof(DIGIT));

    for(j = 0; j < (n2 >> 1); j++)              //ciclo su Vett2, se pari lo finisco
    {
        V2 = _mm_set_epi64x (Vett2[2*j + 1], Vett2[2*j]);       //blocco la prima "coppia" di 64 di Vett2

        printf("V2 = ");
        print__m128(V2);

        for(i = 0; i < (n1 >> 1) ; i++)                         //ciclo su Vett1, lo finisco se è pari
        {
            V1 = _mm_set_epi64x (Vett1[2*i + 1], Vett1[2*i]);   //moltiplico V2 per tutti i blocchi di Vett1

            printf("V1 = ");
            print__m128(V1);

            Res[j+i] ^= clmul(V1, V2);    //salvo in ResTemp

            printf("Clmul Ris = ");
            print__m256(clmul(V1, V2));

            printf("ResTemp[%ld] = ", j+i);
            print__m256(ResTemp[j+i]);
        }

        if(n1 % 2 == 1)  //se Vett1 dispari
        {
            V1 = _mm_set_epi64x (0, Vett1[n1-1]);               //metto l'ultimo blocco di Vett1 in v1, inserendo 0 nella parte "più significativa"
            ResTemp[j+(n1>>1)] ^= clmul(V1, V2);          //salvo in ResTemp l'ultima moltiplicazione
        }
    }

    if(n2 % 2 == 1)                             //se Vett2 dispari
    {
        V2 = _mm_set_epi64x (0, Vett2[n2-1]);                   //blocco l'ultima coppia di Vett2

        for(i = 0; i < (n1 >> 1) ; i++)                         //ciclo su Vett1, se pari lo finisco
        {
            V1 = _mm_set_epi64x (Vett1[2*i + 1], Vett1[2*i]);       //moltiplico V2 per tutti i blocchi di Vett1
            ResTemp[(n2>>1)+i] ^= clmul(V1, V2);  //salvo in ResTemp
        }

        if(n1 % 2 == 1)                                         //se Vett1 dispari
        {
            V1 = _mm_set_epi64x (0, Vett1[n1-1]);               //metto l'ultimo blocco di Vett1 in v1, inserendo 0 nella parte "più significativa"
            ResTemp[lTmp-1] ^= clmul((__m128i)V1, V2);     //salvo in ResTemp l'ultima moltiplicazione
        }
    }

    printf("\nStampo ResTemp dopo costruzione\n");
    for (i = 0; i < lTmp; i++) {
        printf("ResTemp[%ld] = ", i);
        print__m256(ResTemp[i]);
    }

    alignas (32) DIGIT v[4];
    for (j = 0; j < lTmp; j++)      // ciclo su Res256
    {
        _mm256_store_si256((__m256i*)v, ResTemp[j]);     //salvo in memoria Res256[j] e lo associo al vettore v

        for (i = 0; i < 4; i++)         //4 * 64 = 256
            Res[(j*4)+i] = v[i];        //ogni elemento a 64bit di Res256[j] viene salvato in Res con l'offset adeguato

    }

    printf("\nStampo Res dopo costruzione\n");
    for (i = 0; i < n3; i++) {
        printf("Res[%ld] = %016lX\n", i, Res[i]);
    }

}

void Add(uint32_t n3, DIGIT Res[], uint32_t n1, DIGIT Vett1[], uint32_t n2, DIGIT Vett2[])     //Add Vett1 e Vett2 in Res
{
    int32_t i;
    for(i = 0; i < n3; i++)
        Res[i] = Vett1[i] ^ Vett2[i];
}

void ACK(const uint32_t n3, DIGIT Res[], const uint32_t n1, DIGIT Vett1[], const uint32_t n2, DIGIT Vett2[])     //moltiplicazione da dx a sx (0 -> n)
{
    const int32_t a = (n1+1) >> 1;                            //lunghezza dei vettori A0 e A1
    const int32_t b = (n2+1) >> 1;                                //lunghezza dei vettori B0 e B1
    const uint32_t l = a + b;                          //lunghezza dei vettori risultato di Array_Clmul
    int32_t i = 0, j = 0;
    DIGIT A0[MAX_SIZE], A1[MAX_SIZE], B0[MAX_SIZE], B1[MAX_SIZE];    //A0 parte meno significativa, riempita con il bit medio. A1 parte più significativa, riempita con uno zero.
    DIGIT SumA[a], SumB[b];              //temporanei per gli xor A0 + A1, B0 + B1
    DIGIT C[MAX_SIZE], D[MAX_SIZE], E[MAX_SIZE];             //temporanei per l'uscita da Array Clmul

    memset(Res, 0x00, n3*sizeof(DIGIT));

    // A0 <= da Vett1[0] a Vett[n1/2] (se dispari n1+1)

    for(i = 0; i < (n1 >> 1); i++)          //riempio A0 e A1, ignorando il bit "in mezzo" se dispari
    {
        A0[i] = Vett1[i];
        A1[i] = Vett1[a+i];
    }

    if(n1 % 2 == (uint32_t)1)               //se n1 dispari
    {
        A0[a-1] = Vett1[a-1];               //bit finale di A0
        A1[a-1] = (DIGIT) 0;             //bit finale di A1
    }

    for(i = 0; i < (n2 >> 1); i++)          //riempio B0 e B1, ignorando il bit "in mezzo" se dispari
    {
        B0[i] = Vett2[i];
        B1[i] = Vett2[b+i];
    }

    if(n2 % 2 == (uint32_t)1)               //se n2 dispari
    {
        B0[b-1] = Vett2[b-1];               //bit finale di B0
        B1[b-1] = (DIGIT) 0;             //bit finale di B1
    }


    Array_Clmul(l, C, a, A1, b, B1);        //A1 * B1 ottengo c1, c0
    Array_Clmul(l, D, a, A0, b, B0);        //A0 * B0 ottengo d1, d0

    Add(a, SumA, a, A0, a, A1);             //A0 + A1
    Add(b, SumB, b, B0, b, B1);             //B0 + B1

    printf("\n\nGuardo E\n");
    Array_Clmul(l, E, a, SumA, b, SumB);    //(A0 + A1) * (B0 + B1)

    printf("\nStampo A0:\n");
    for (i = 0; i < a; i++)
        printf("A0[%d] = %016lX\n", i, A0[i]);

    printf("\nStampo A1:\n");
    for (i = 0; i < a; i++)
        printf("A1[%d] = %016lX\n", i, A1[i]);

    printf("\nStampo B0:\n");
    for (i = 0; i < b; i++)
        printf("B0[%d] = %016lX\n", i, B0[i]);

    printf("\nStampo B1:\n");
    for (i = 0; i < b; i++)
        printf("B1[%d] = %016lX\n", i, B1[i]);

    printf("\nStampo C:\n");
    for (i = 0; i < l; i++)
        printf("C[%d] = %016lX\n", i, C[i]);

    printf("\nStampo D:\n");
    for (i = 0; i < l; i++)
        printf("D[%d] = %016lX\n", i, D[i]);

    printf("\nStampo sumA:\n");
    for (i = 0; i < a; i++)
        printf("sumA[%d] = %016lX\n", i, SumA[i]);

    printf("\nStampo sumB:\n");
    for (i = 0; i < b; i++)
        printf("sumB[%d] = %016lX\n", i, SumB[i]);

    printf("\nStampo E:\n");
    for (i = 0; i < l; i++)
        printf("E[%d] = %016lX\n", i, E[i]);

    printf("\n l = %d\n", l);
    printf("\n a = %d\n", a);
    printf("\n b = %d\n", b);

    // ricostruisce il vettore finale con karatsuba
    for (i = 0, j = l-1; i < l; i++, j--) {
        Res[i] ^= C[j];
        Res[i + (l>>1)] ^= D[j] ^ C[j] ^ E[j];
        Res[i + (n1 >> 1) + (n2 >> 1)] ^= D[j];
    }

    /*if (n1 % 2 == 1) {
        for (i = 0; i < n3-3; i++)
            Res[i] = Res[i+2];

        Res[n3-2] = (DIGIT) 0;
        Res[n3-1] = (DIGIT) 0;
    }*/
}
