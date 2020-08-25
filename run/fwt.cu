#include <stdio.h>

__global__ void FWT(float *fi, float *Fa, float *seq, const int Pa,
        const int Na, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // thread Id
    int lid = tid % 32;                                 // lane Id
    int bid = blockIdx.x;                               // block Id

    float F1; // storing last value
    float F2; // will be shuffled, all threads have one

    float seqi; // where in memory to put value

    // calculate whether mem pull will be made neg
    // [0:1] -> [0:2] -> [-1:1] -> [1:-1]
    // int negMask = (((tid >> 0) & 1LU) * 2 - 1) * -1;
    int negMask;

    // whether to accept shfl this round
    int srcMask;

    if (tid < N) {
        F1 = fi[tid];
    }

    // trying to hide memory pull with ops
    seqi = seq[tid];

    Nm = 1;
    for(int pm=0;pm<Pa;pm++) {
        // calculate negMask
        negMask = (((tid >> pm) & 1LU) * 2 - 1) * -1;    // 1 or -1

        // calculate src mask
        srcMask = (tid >> pm) & 1LU; // 0 or 1

        // apply warp shuffle down
        F2 = srcMask * __shfl_down_sync(0xFFFFFFFF, F1, Nm);

        // flip mask
        srcMask ^= 1LU;

        // apply warp shuffle up
        F2 += srcMask * __shfl_up_sync(0xFFFFFFFF, F1, Nm);

        // add to existing warp value, using negMask
        F1 = F1 * negMask + F2;

        // update shfl width
        Nm <<= 1
    }

    // write to global memory
    Fa[(tid / 32) * 32 + seqi] = F1;
}

extern "C"
{

// int main(int argc, char *argv)
void run_FWT(const int Pa, const int Na, const int N,
        float *fi, float *Fa, float *seq, const int blockDimX,
        const int gridDimX)
{
    // each individual FWT size
    // const int Pa = 3;
    // const int Na = pow(Pa, 2);

    // total array of data size
    // const int N;
    // float *fi; // input
    // float *Fa; // output

    // sequency mapping
    // float *seq;

    dim3 blockSize = (blockDimX, 1, 1);
    dim3 gridSize  = ( gridDimX, 1, 1);
    FHT<<<gridSize, blockSize>>>(fi, Fa, seq, Pa, Na, N);
}

}
