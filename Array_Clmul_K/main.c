#include "Clmul.h"

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>

void Add(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[]);

int main(int argc, char *argv[]) {
    uint32_t n1, n2, n3;
    uint64_t Vett1[n1], Vett2[n2], Res[n3];
    n1 = n2 = n3 = 1;
    Vett1[0] = UINT64_MAX;
    Vett2[0] = UINT64_MAX;
    Res[0] = 1;
    Add(n3, Res, n1, Vett1, n2, Vett2);
    printf("%04X", Res[0]);
    return 0;
}

void Add(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[]){    //Add
    int i;
    for(i = 0; i < n3; i++)
    asm(
    "movq %[Vett1], %%rax\n\t"  // muovo Vett1 nel registro rax
    "movq %[Vett2], %%rbx\n\t"  // muovo Vett2 nel registro rbx
    "xor %%rbx, %%rax\n\t"      // Xor tra rax(Vett1) e rbx(Vett2) salvato in rax
    "movq %%rax, %[Res]\n\t"    // muovo rax in Res
    : [Res] "+rm" (Res[i])
    : [Vett1] "rm" (Vett1[i]), [Vett2] "rm" (Vett2[i])
    );

return;
}

void ACK(uint32_t n3, uint64_t Res[], uint32_t n1, uint64_t Vett1[], uint32_t n2, uint64_t Vett2[]){    //moltiplicazione da dx a sx (0 -> n)
    int l1, l2 ,l3;         // lunghezze vettori risultato di Array_Clmul
    __m256i A0[(n1+1) >> 1], A1[(n1) >> 1], B0[(n2+1) >> 1], B1[(n2+1)>>1];   //A0 parte meno significativa più lunga se dispari
    __m256i Res1[l1], Res2[l2], Res3[l3];
        //    A0 <= da Vett1[0] a Vett[n1/2] (se dispari n1+1)
}
