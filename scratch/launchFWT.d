extern (C) void run_FWT(const int Pa, const int Na, const int N,
        float *fi, float *Fa, float *seq, const int blockDimX,
        const int gridDimX)

import std.stdio;
import std.math: sin;

import cuda_d.cuda;
import cuda_d.cuda_runtime_api;
import cuda_d.cublas_api;

void main()
{
    const int numFWTs = 10;
    const int Pa = 4;
    const int Na = 2^^Pa;
    const int N  = numFWTs * N;

    // calculate sequency mapping (same for each FWT)
    // give each their own array? or let read clash happen... (TODO)
    int[] s;
    s.length = Na;
    for(int i=0;i<Na;i++) {
        s[i] = i;
    }
    auto k = getSequence(s, Na, P);

    // generate fake fi data, and empty Fa data
    // (real data given by discrete function data (fi),
    //  modified as such:
    //   f = fi * dx / sqrt(xb - xa) ... (TODO) )
    float[] fi, Fa;
    fi.length = N;
    Fa.length = N;
    for(int i=1;i<N;i++) {
        fi[i] = sin(i*0.05);
    }

    // set-up device side data
    float* d_fi, d_Fa, d_seq; // D-style pointers
    const ulong numBytes = N * float.sizeof;
    cudaMalloc( cast(void**)&d_fi , numBytes );
    cudaMalloc( cast(void**)&d_Fa , numBytes );
    cudaMalloc( cast(void**)&d_seq, numBytes );
    
    /// copy data from host to device
    cudaMemcpy( cast(void*)d_fi , cast(void*)fi, numBytes,
            cudaMemcpyKind.cudaMemcpyHostToDevice );
    cudaMemcpy( cast(void*)d_seq, cast(void*)k , numBytes,
            cudaMemcpyKind.cudaMemcpyHostToDevice );

    // calculate cuda blocks / grid required
    int blockDimX = 32;
    int gridDimX  = (N + 32 - 1) / 32;

    // run kernel
    run_FWT(Pa, Na, N, fi, Fa, seq, blockDimX, gridDimX);

    // copy data from device to host
    cudaMemcpy( cast(void*)Fa, cast(void*)d_Fa, Mintsize,
            cudaMemcpyKind.cudaMemcpyDeviceToHost );

    // free memory on device
    cudaFree( d_fi );
    cudaFree( d_Fa );
    cudaFree( d_seq );

    return;
}

int[] getSequence(int[] s, int N, int P)
{
    int[] g, k;
    g.length = N;
    k.length = N;

    for (int i=0;i<N;i++) {
        g[i] = s[i] ^ (s[i] >> 1);
    }

    for (int i=0;i<N;i++) {
        k[i] = bitReverse(g[i], P);
    }

    return k;
}

int bitReverse(int i, int size)
{
    import std.algorithm.mutation: reverse;
    import std.format: format;
    import std.conv: to, parse;

    string fmt = format!"%%0%db"(size);
    auto ss = format(fmt, i);
    string st = ss.dup().reverse;
    int pst = parse!int(st, 2);
    return pst;
}
