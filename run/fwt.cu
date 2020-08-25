#include <stdio.h>

__global__ void FWT(float *fi, float *Fa, int *seq, const int Pa,
        const int Na, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // thread Id

    float F1; // storing last value
    float F2; // will be shuffled, all threads have one

    int seqi; // where in memory to put value

    // calculate whether mem pull will be made neg
    // [0:1] -> [0:2] -> [-1:1] -> [1:-1]
    // int negMask = (((tid >> 0) & 1LU) * 2 - 1) * -1;
    int negMask;

    // whether to accept shfl this round
    int srcMask;

    // trying to hide memory pull with ops (not anymore lol)
    seqi = seq[tid];

    if (tid < N) {
        // F1 = fi[tid];
        F1 = fi[(tid / 32) * 32 + seqi];
    }

    int Nm = Na/2;
    for(int pm=0;pm<Pa;pm++) {
        // calculate negMask
        negMask = (((tid >> (Pa-pm-1)) & 1LU) * 2 - 1) * -1;    // 1 or -1

        // calculate src mask
        srcMask = ((tid >> (Pa-pm-1)) & 1LU) ^ 1LU; // 0 or 1
        // if (tid == 2) {
            // printf("tid:%d, pm=%d, srcMask=%d, Nm=%d, negMask=%d\n",
                    // tid, pm, srcMask, Nm, negMask);
        // }

        // apply warp shuffle down
        F2 = srcMask * __shfl_down_sync(0xFFFFFFFF, F1, Nm);

        // flip mask
        srcMask ^= 1LU;
        // if (tid == 2) {
            // printf("tid:%d, pm=%d, srcMask=%d, Nm=%d, negMask=%d\n",
                    // tid, pm, srcMask, Nm, negMask);
        // }

        // if (tid == 7) {
            // printf("F1=%f, F2=%f\n", F1, F2);
        // }

        // apply warp shuffle up
        F2 += srcMask * __shfl_up_sync(0xFFFFFFFF, F1, Nm);

        // if (tid == 2) {
            // printf("F1=%f, F2=%f\n", F1, F2);
        // }

        // add to existing warp value, using negMask
        F1 = F1 * negMask + F2;

        // update shfl width
        // Nm <<= 1;
        Nm >>= 1;
    }

    // write to global memory
    if (tid < N) {
        // Fa[(tid / 32) * 32 + seqi] = F1;
        Fa[tid] = F1;
    }

    return;
}

extern "C"
{
    void run_FWT(const int Pa, const int Na, const int N,
            float *fi, float *Fa, int *seq, const int blockDimX,
            const int gridDimX)
    {
        dim3 blockSize(blockDimX, 1, 1);
        dim3 gridSize( gridDimX, 1, 1);

        FWT<<<gridSize, blockSize>>>(fi, Fa, seq, Pa, Na, N);

        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(err));
        }
    }
}
