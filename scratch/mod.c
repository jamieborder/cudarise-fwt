#include <stdio.h>

int main(int argc, char *argv)
{
    int tid = 14;
    int Nm = 16;

    printf("tid=%d, Nm=%d, idx=%d\n", tid, Nm, (tid - Nm + 32 + 32) % 32);
}
