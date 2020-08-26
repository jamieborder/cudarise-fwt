#include <stdio.h>

__global__ void FWT_SHFL(const float *fi, float *Fa, const int *seq,
        const int Pa, const int Na, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // thread Id

    float F1; // storing last value
    float F2; // will be shuffled, all threads have one

    int seqi; // where in memory to put value

    // calculate whether mem pull will be made neg
    // [0:1] -> [0:2] -> [-1:1] -> [1:-1]
    int negMask;

    // whether to accept shfl this round
    int srcMask;

    // every thread load first piece of data
    if (tid < N) {
        F1 = fi[tid];
    }

    // memory pull hidden by next ops
    seqi = seq[tid];

    int Nm = Na/2;
    for(int pm=0;pm<Pa;pm++) {
        // calculate negMask
        negMask = (((tid >> (Pa-pm-1)) & 1LU) * 2 - 1) * -1;    // 1 or -1

        // calculate src mask
        srcMask = ((tid >> (Pa-pm-1)) & 1LU) ^ 1LU; // 0 or 1

        // apply warp shuffle down
        F2 = srcMask * __shfl_down_sync(0xFFFFFFFF, F1, Nm);

        // flip mask
        srcMask ^= 1LU;

        // apply warp shuffle up
        F2 += srcMask * __shfl_up_sync(0xFFFFFFFF, F1, Nm);

        // add to existing warp value, using negMask
        F1 = F1 * negMask + F2;

        // update shfl width
        Nm >>= 1;
    }

    // write to global memory
    if (tid < N) {
        Fa[(tid / 32) * 32 + seqi] = F1;
    }

    return;
}

// global memory version
__global__ void FWT_GM(const float *fi, float *Fa, const int *seq,
        const int Pa, const int Na, const int N, float *F1_global)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // thread Id
    int lid = tid % 32;

    float F1; // storing last value
    float F2; // will be shuffled, all threads have one

    int seqi; // where in memory to put value

    // calculate whether mem pull will be made neg
    // [0:1] -> [0:2] -> [-1:1] -> [1:-1]
    int negMask;

    // whether to accept shfl this round
    int srcMask;

    // memory pull hidden by next ops
    seqi = seq[tid];

    // every thread load first piece of data
    if (tid < N) {
        F1 = fi[tid];
    }

    // now write all data to global memory so all threads can get it
    F1_global[tid] = F1;

    int Nm = Na/2;
    for(int pm=0;pm<Pa;pm++) {
        // calculate negMask
        negMask = (((tid >> (Pa-pm-1)) & 1LU) * 2 - 1) * -1;    // 1 or -1

        // calculate src mask
        srcMask = ((tid >> (Pa-pm-1)) & 1LU) ^ 1LU; // 0 or 1

        // apply warp shuffle down
        F2 = srcMask * F1_global[(tid / 32) * 32 + (lid + Nm) % 32];

        // flip mask
        srcMask ^= 1LU;

        // apply warp shuffle up
        F2 += srcMask * F1_global[(tid / 32) * 32 + (lid - Nm) % 32];

        // add to existing warp value, using negMask
        F1 = F1 * negMask + F2;

        // and write to global memory
        F1_global[tid] = F1;

        // update shfl width
        Nm >>= 1;
    }

    // write to global memory
    if (tid < N) {
        Fa[(tid / 32) * 32 + seqi] = F1;
    }

    return;
}

// shared memory version
__global__ void FWT_SM(const float *fi, float *Fa, const int *seq,
        const int Pa, const int Na, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // thread Id
    int lid = tid % 32;

    extern __shared__ float F1_shared[];

    float F1; // storing last value
    float F2; // will be shuffled, all threads have one

    int seqi; // where in memory to put value

    // calculate whether mem pull will be made neg
    // [0:1] -> [0:2] -> [-1:1] -> [1:-1]
    int negMask;

    // whether to accept shfl this round
    int srcMask;

    // memory pull hidden by next ops
    seqi = seq[tid];

    // every thread load first piece of data
    if (tid < N) {
        F1 = fi[tid];
    }

    // now write all data to global memory so all threads can get it
    F1_shared[lid] = F1;

    int Nm = Na/2;
    for(int pm=0;pm<Pa;pm++) {
        // calculate negMask
        negMask = (((tid >> (Pa-pm-1)) & 1LU) * 2 - 1) * -1;    // 1 or -1

        // calculate src mask
        srcMask = ((tid >> (Pa-pm-1)) & 1LU) ^ 1LU; // 0 or 1

        // apply warp shuffle down
        F2 = srcMask * F1_shared[(lid + Nm) % 32];

        // flip mask
        srcMask ^= 1LU;

        // apply warp shuffle up
        F2 += srcMask * F1_shared[(lid - Nm + 32) % 32];

        // add to existing warp value, using negMask
        F1 = F1 * negMask + F2;

        // and write to global memory
        F1_shared[lid] = F1;

        // update shfl width
        Nm >>= 1;
    }

    // write to global memory
    if (tid < N) {
        Fa[(tid / 32) * 32 + seqi] = F1;
    }

    return;
}

extern "C"
{
    void run_FWT_SHFL(const int Pa, const int Na, const int N, const float *fi,
            float *Fa, const int *seq, const int blockDimX, const int gridDimX)
    {
        dim3 blockSize(blockDimX, 1, 1);
        dim3 gridSize( gridDimX, 1, 1);

        FWT_SHFL<<<gridSize, blockSize>>>(fi, Fa, seq, Pa, Na, N);

        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(err));
        }
    }

    void run_FWT_GM(const int Pa, const int Na, const int N, const float *fi,
            float *Fa, const int *seq, float *vec, const int blockDimX,
            const int gridDimX)
    {
        dim3 blockSize(blockDimX, 1, 1);
        dim3 gridSize( gridDimX, 1, 1);

        FWT_GM<<<gridSize, blockSize>>>(fi, Fa, seq, Pa, Na, N, vec);

        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(err));
        }
    }

    void run_FWT_SM(const int Pa, const int Na, const int N, const float *fi,
            float *Fa, const int *seq, const int blockDimX, const int gridDimX,
            const int sMemSize)
    {
        dim3 blockSize(blockDimX, 1, 1);
        dim3 gridSize( gridDimX, 1, 1);

        FWT_SM<<<gridSize, blockSize, sMemSize>>>(fi, Fa, seq, Pa, Na, N);

        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(err));
        }
    }
}
