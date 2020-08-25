extern (C) void run_FWT(const int Pa, const int Na, const int N,
        float *fi, float *Fa, int *seq, const int blockDimX,
        const int gridDimX);

import std.stdio;
import std.math: sin;
import std.datetime: MonoTime;
import std.conv: to;

import cuda_d.cuda;
import cuda_d.cuda_runtime_api;
import cuda_d.cublas_api;

void main()
{
    // const int numFWTs = 1000000;
    const int numFWTs = 1;
    const int Pa = 5;
    const int Na = 2^^Pa;
    const int N  = numFWTs * Na;

    writefln("number of FWTs :: %d", numFWTs);
    writefln("Pa = %d, Na = %d, N = %d", Pa, Na, N);
    writefln("number of cells :: %d", Na / 2 * numFWTs);

    // calculate sequency mapping (same for each FWT)
    // give each their own array? or let read clash happen... (TODO)
    int[] s;
    s.length = Na;
    for(int i=0;i<Na;i++) {
        s[i] = i;
    }
    auto k = getSequence(s, Na, Pa);

    int[] seq;
    seq.length = N;
    for(int i=0;i<N;i+=Na) {
        seq[i..i+Na] = k[];
    }

    // generate fake fi data, and empty Fa data
    // (real data given by discrete function data (fi),
    //  modified as such:
    //   f = fi * dx / sqrt(xb - xa) ... (TODO) )
    float[] fi, Fa;
    fi.length = N;
    Fa.length = N;
    for(int i=0;i<N;i++) {
        // fi[i] = 0.0;
        // fi[i] = 1.0;
        fi[i] = i;
        // fi[i] = sin(i*0.05);
    }

    // set-up device side data
    float* d_fi, d_Fa; // D-style pointers
    int* d_seq;
    const ulong numBytesFloat = N * float.sizeof;
    const ulong numBytesInt   = N * int.sizeof;
    cudaMalloc( cast(void**)&d_fi , numBytesFloat );
    cudaMalloc( cast(void**)&d_Fa , numBytesFloat );
    cudaMalloc( cast(void**)&d_seq, numBytesInt );
    
    /// copy data from host to device
    cudaMemcpy( cast(void*)d_fi , cast(void*)fi, numBytesFloat,
            cudaMemcpyKind.cudaMemcpyHostToDevice );
    cudaMemcpy( cast(void*)d_seq, cast(void*)seq, numBytesInt,
            cudaMemcpyKind.cudaMemcpyHostToDevice );

    // calculate cuda blocks / grid required
    int blockDimX = 32;
    int gridDimX  = (N + 32 - 1) / 32;

    writefln("launch with blockDim = (%d, 1, 1)", blockDimX);
    writefln("launch with  gridDim = (%d, 1, 1)", gridDimX);

    auto startTime = MonoTime.currTime;

    // run kernel
    run_FWT(Pa, Na, N, d_fi, d_Fa, d_seq, blockDimX, gridDimX);

    // deviceSynchronize on C side
    auto gpuTime = MonoTime.currTime - startTime;

    // copy data from device to host
    cudaMemcpy( cast(void*)Fa, cast(void*)d_Fa, numBytesFloat,
            cudaMemcpyKind.cudaMemcpyDeviceToHost );

    // free memory on device
    cudaFree( d_fi );
    cudaFree( d_Fa );
    cudaFree( d_seq );

    float[] Fa_cpu;
    Fa_cpu.length = N;

    startTime = MonoTime.currTime;
    for(int i=0;i<N;i+=Na) {
        Fa_cpu[i..i+Na] = FWT_fiToFaFromPoints(Pa, Na,
            0.0, 1.0, fi[i..i+Na]);
    }
    auto cpuTime = MonoTime.currTime - startTime;

    writeln("Fa_gpu = \n", Fa);
    writeln("time for GPU: ", gpuTime);

    writeln("Fa_cpu = \n", Fa_cpu);
    writeln("time for CPU: ", cpuTime);

    writeln("GPU speed-up: ", cpuTime.total!"usecs"
            / gpuTime.total!"usecs");

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

// If using an extended domain, probably need to adjust accordingly - change d_x,
//  x_a, x_b, x_midPoints.
float[] FWT_fiToFaFromPoints(in size_t p_a, in size_t N_a,
        in float x_a, in float x_b, float[] fiCopy)//fiAtMidPoints)
{
    // size_t N_a = 2^^p_a;
    // float d_x = (x_b - x_a) / N_a;
    float[] F_currBlock, F_prevBlock;
    F_currBlock.length = N_a;
    F_prevBlock.length = N_a;

    // float[] fiCopy = fiAtMidPoints.dup();

    // foreach(ref fi; fiCopy) {
        // fi = fi * d_x / sqrt(x_b - x_a);
    // }

    size_t i1, i2, i3;
    // Block 1.
    foreach(a; 1 .. N_a / 2 + 1) {
        i1 = 2 * a - 1;
        i2 = i1 + 1;

        F_currBlock[i1 - 1] = fiCopy[i1 - 1] + fiCopy[i2 - 1];
        F_currBlock[i2 - 1] = fiCopy[i1 - 1] - fiCopy[i2 - 1];
    }
    F_prevBlock = F_currBlock.dup();
    // Block(s) 2 --> p_a.
    foreach(p_m; 2 .. p_a + 1) {
        size_t N_pm = 2^^p_m;
        foreach(G; 1 .. N_a / N_pm + 1) {
            // solving top
            foreach(a; 1 .. N_pm / 2 + 1) {
                i1 = a + (G - 1) * N_pm;
                i2 = 2 * a - 1 + (G - 1) * N_pm;
                i3 = i2 + 1;
                
                F_currBlock[i2 - 1] = F_prevBlock[i1 - 1];
                F_currBlock[i3 - 1] = F_prevBlock[i1 - 1];
            }
            // solving bottom
            foreach(a; 1 .. N_pm / 2 + 1) {
                i1 = N_pm + 1 - a + (G - 1) * N_pm;
                i2 = N_pm + 2 - 2 * a + (G - 1) * N_pm;
                i3 = i2 - 1;
                long theta = to!long((-1.)^^(a + 1));

                F_currBlock[i2 - 1] = F_currBlock[i2 - 1] + theta * F_prevBlock[i1 - 1];
                F_currBlock[i3 - 1] = F_currBlock[i3 - 1] - theta * F_prevBlock[i1 - 1];
            }
        }
        F_prevBlock = F_currBlock.dup();
    }
    return F_currBlock;
}
