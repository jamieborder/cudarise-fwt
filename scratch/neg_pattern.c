#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void negPat(int *s, const int N, const int iter)
{
    for (int i=0;i<iter;i++) {
        for (int tid=0;tid<N;tid++) {
            // if ((tid >> i) & 1LU) {
                // s[tid] = -1;
            // } 
            s[tid] = (((tid >> i) & 1LU) * 2 - 1) * - 1;
        }

        for (int l=0;l<N;l++) {
            printf("% d ", s[l]);
            s[l] = 0;
        }
        printf("\n");
    }
}

int main(int argc, char *argv)
{
    const int PPP = 3;
    const int NNN = pow(PPP, 2);
    printf("%d %d\n", PPP, NNN);

    for (int ii=0;ii<64;ii++) {
        printf("%d %d\n", ii, (ii / 32) * 32);
    }

    int o = 1;
    printf("%d\n",o);
    for (int f=0;f<10;f++) {
        o ^= 1LU;
        printf("%d\n",o);
    }

    printf("hello world\n");

    int P = 3;
    int N = pow(2,P);

    int *s;
    s = (int *)malloc(sizeof(int)*N); 

    long int i = 0b10111;

    printf("%ld\n", i);

    int isBitSet;
    for (int j=0;j<sizeof(i);j++) {
        isBitSet = (i >> j) & 1LU;

        printf("is bit %d set ? %d\n", j, isBitSet);
    }
    

    negPat(s, N, 3);
}
