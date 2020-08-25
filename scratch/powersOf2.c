#include <stdio.h>

int main(int argc, char *argv)
{
    int Nm = 1;
    printf("%d\n", Nm);
    for(int i=0;i<4;i++) {
        Nm <<= 1;
        printf("%d\n", Nm);
    }

    return 0;
}
