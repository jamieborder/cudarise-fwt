extern (C) void run_FWT_SHFL(const int Pa, const int Na, const int N,
        const float *fi, float *Fa, const int *seq, const int blockDimX,
        const int gridDimX);

extern (C) void run_FWT_GM(const int Pa, const int Na, const int N,
        const float *fi, float *Fa, const int *seq, float *vec,
        const int blockDimX, const int gridDimX);

extern (C) void run_FWT_SM(const int Pa, const int Na, const int N,
        const float *fi, float *Fa, const int *seq, const int blockDimX,
        const int gridDimX, const int sMemSize);

extern(C) void run_FWT_SIMD(const int Pa, const int Na, const int N,
        const float *fi, float *Fa, const int *seq);

import std.stdio;
import std.math: sin;
import std.datetime: MonoTime;
import std.conv: to;

import cuda_d.cuda;
import cuda_d.cuda_runtime_api;
import cuda_d.cublas_api;

void main(string[] args)
{
    int numFWTs = 1;
    int Pa = 5;
    int verb = 0;
    int numRuns = 10;

    immutable string helpStr =
        "./prog [numFWTs, [Pa, [verbosity, [numRuns]]]]";
    if (args.length > 1) {
        scope(failure) writeln(helpStr);
        numFWTs = to!int(args[1]);
        if (args.length > 2) {
            Pa = to!int(args[2]);
            if (args.length > 3) {
                verb = to!int(args[3]);
                if (args.length > 4) {
                    numRuns = to!int(args[4]);
                }
            }
        }
    }

    int Na = 2^^Pa;
    int N  = numFWTs * Na;

    long[5] times;
    long[5] avgs = [0, 0, 0, 0, 0];
    for (int i=0;i<numRuns;i++) {
        launchTimings(numFWTs, Pa, Na, N, verb, times);
        avgs[] += times[];
    }
    avgs[] = avgs[] / numRuns; 
    writefln("   numFWTs     cpu gpu_SHFL  gpu_GM  gpu_SM cpu_SIMD (ms)");
    writefln("%10d %7.3f %7.3f  %7.3f %7.3f %7.3f", numFWTs, avgs[0]/1000.,
            avgs[1]/1000., avgs[2]/1000., avgs[3]/1000., avgs[4]/1000.);
}

void launchTimings(const int numFWTs, const int Pa, const int Na,
        const int N, const int verb, ref long[5] times)
{
    if (verb > 0) {
        writefln(" ~~ CUDA Accelerated Fast Walsh Transform ~~ ");
        writefln("\tnumber of FWTs = %d", numFWTs);
        writefln("\tPa = %d, Na = %d, N = %d", Pa, Na, N);
        writefln("\tnumber of cells = %d", Na / 2 * numFWTs);
        writefln("\tverbosity = %d", verb);
    }

    // calculate sequency mapping (same for each FWT)
    // give each their own array? or let read clash happen... (TODO)
    int[] s;
    s.length = Na;
    for(int i=0;i<Na;i++) {
        s[i] = i;
    }
    auto k = getSequence(s, Na, Pa);
    if (verb > 1) {
        writeln("k = \n", k);
    }

    int[] seq;
    seq.length = N;
    for(int i=0;i<N;i+=Na) {
        seq[i..i+Na] = k[];
    }

    // generate fake fi data, and empty Fa data
    // (real data given by discrete function data (fi),
    //  modified as such:
    //   f = fi * dx / sqrt(xb - xa) ... (TODO) )
    float[] fi, Fa_SHFL, Fa_GM, Fa_SM;
    fi.length = N;
    Fa_SHFL.length = N;
    Fa_GM.length = N;
    Fa_SM.length = N;
    for(int i=0;i<N;i++) {
        fi[i] = sin(i*0.05);
    }

    // calculate cuda blocks / grid required
    int blockDimX = 32;
    // int blockDimX = 64;
    // int blockDimX = 128;
    int gridDimX  = (N + blockDimX - 1) / blockDimX;

    if (verb > 1) {
        writefln("\tlaunch with blockDim = (%d, 1, 1)", blockDimX);
        writefln("\tlaunch with  gridDim = (%d, 1, 1)", gridDimX);
    }

    // set-up device side data
    float* d_fi, d_Fa; // D-style pointers
    int* d_seq;
    const ulong numBytesFloat = N * float.sizeof;
    const ulong numBytesInt   = N * int.sizeof;
    cudaMalloc( cast(void**)&d_fi , numBytesFloat );
    cudaMalloc( cast(void**)&d_Fa , numBytesFloat );
    cudaMalloc( cast(void**)&d_seq, numBytesInt );

    // device side work array for GM kernel
    float* d_vec;
    cudaMalloc( cast(void**)&d_vec, numBytesFloat );
    
    /// copy data from host to device
    cudaMemcpy( cast(void*)d_fi , cast(void*)fi, numBytesFloat,
            cudaMemcpyKind.cudaMemcpyHostToDevice );
    cudaMemcpy( cast(void*)d_seq, cast(void*)seq, numBytesInt,
            cudaMemcpyKind.cudaMemcpyHostToDevice );



    /// ----- shfls -----
    auto startTime = MonoTime.currTime;

    // run kernel
    run_FWT_SHFL(Pa, Na, N, d_fi, d_Fa, d_seq, blockDimX, gridDimX);

    // deviceSynchronize on C side
    auto gpuTime = MonoTime.currTime - startTime;

    // copy data from device to host
    cudaMemcpy( cast(void*)Fa_SHFL, cast(void*)d_Fa, numBytesFloat,
            cudaMemcpyKind.cudaMemcpyDeviceToHost );



    /// ----- global memory -----
    startTime = MonoTime.currTime;

    // run kernel
    run_FWT_GM(Pa, Na, N, d_fi, d_Fa, d_seq, d_vec, blockDimX, gridDimX);

    // deviceSynchronize on C side
    auto gpuTime_GM = MonoTime.currTime - startTime;

    // copy data from device to host
    cudaMemcpy( cast(void*)Fa_GM, cast(void*)d_Fa, numBytesFloat,
            cudaMemcpyKind.cudaMemcpyDeviceToHost );



    /// ----- shared memory -----
    int sMemSize = to!int(Na * float.sizeof);
    startTime = MonoTime.currTime;

    // run kernel
    run_FWT_SM(Pa, Na, N, d_fi, d_Fa, d_seq, blockDimX, gridDimX,
            sMemSize);

    // deviceSynchronize on C side
    auto gpuTime_SM = MonoTime.currTime - startTime;

    // copy data from device to host
    cudaMemcpy( cast(void*)Fa_SM, cast(void*)d_Fa, numBytesFloat,
            cudaMemcpyKind.cudaMemcpyDeviceToHost );



    // free memory on device
    cudaFree( d_fi );
    cudaFree( d_Fa );
    cudaFree( d_seq );
    cudaFree( d_vec );

    /*
    for(int i=0;i<Na;++i) {
        fi[i] = i * 1.0f;
    }
    */

    /// ----- cpu -----
    float[] Fa_cpu;
    Fa_cpu.length = N;

    startTime = MonoTime.currTime;
    for(int i=0;i<N;i+=Na) {
        Fa_cpu[i..i+Na] = FWT_fiToFaFromPoints(Pa, Na,
            0.0, 1.0, fi[i..i+Na]);
    }
    auto cpuTime = MonoTime.currTime - startTime;

    /*
    for (int i=0;i<Na;++i) {
        printf("cpu : %d : %f\n", i, Fa_cpu[i]);
    }
    */

    /// ----- cpu simd -----
    float[] Fa_simd;
    Fa_simd.length = N;

    if (Na != 8) {
        printf("WARNING! CPU_SIMD not designed for Na!=8. Skipping...\n");
        startTime = MonoTime.currTime;
    }
    else {
        printf("WARNING! CPU_SIMD result is ordered incorrectly...\n");
        startTime = MonoTime.currTime;
        run_FWT_SIMD(Pa, Na, N, &(fi[0]), &(Fa_simd[0]), &(seq[0]));
    }
    auto simdTime = MonoTime.currTime - startTime;

    /*
    for (int i=0;i<Na;++i) {
        printf("simd: %d : %f\n", i, Fa_simd[i]);
    }
    */


    if (verb > 1) {
        writeln("Fa_gpu_SHFL = \n", Fa_SHFL);
        writeln("Fa_gpu_GM   = \n", Fa_GM);
        writeln("Fa_gpu_SM   = \n", Fa_SM);
        writeln("Fa_cpu      = \n", Fa_cpu);
        writeln("Fa_simd     = \n", Fa_simd);
    }

    if (verb > 0) {
        writeln("\ttime for GPU (shfls): ", gpuTime);
        writeln("\ttime for GPU (G mem): ", gpuTime_GM);
        writeln("\ttime for GPU (S mem): ", gpuTime_SM);
        writeln("\ttime for CPU        : ", cpuTime);
        writeln("\ttime for CPU (SIMD) : ", simdTime);
        writeln();

        writeln("\tGPU (shfls) speed-up: ", cpuTime.total!"usecs"
                / gpuTime.total!"usecs");
        writeln("\tGPU (G mem) speed-up: ", cpuTime.total!"usecs"
                / gpuTime_GM.total!"usecs");
        writeln("\tGPU (S mem) speed-up: ", cpuTime.total!"usecs"
                / gpuTime_SM.total!"usecs");
        if (Na == 8) {
            writeln("\tCPU (SIMD) speed-up : ", cpuTime.total!"usecs"
                    / simdTime.total!"usecs");
        }
        writeln();

        import std.algorithm.iteration: map, fold;

        float[] diff;
        diff.length = N;

        diff[] = Fa_SHFL[] - Fa_cpu[];
        writeln("\tL2 error for GPU (shfls): ",
                fold!((a, b) => a + b)(map!(a => a^^2.0)(diff)));
        diff[] = Fa_GM[] - Fa_cpu[];
        writeln("\tL2 error for GPU (G mem): ",
                fold!((a, b) => a + b)(map!(a => a^^2.0)(diff)));
        diff[] = Fa_SM[] - Fa_cpu[];
        writeln("\tL2 error for GPU (S mem): ",
                fold!((a, b) => a + b)(map!(a => a^^2.0)(diff)));
        diff[] = Fa_simd[] - Fa_cpu[];
        writeln("\tL2 error for CPU (SIMD) : ",
                fold!((a, b) => a + b)(map!(a => a^^2.0)(diff)));
    }

    times[0] = cpuTime.total!"usecs";
    times[1] = gpuTime.total!"usecs";
    times[2] = gpuTime_GM.total!"usecs";
    times[3] = gpuTime_SM.total!"usecs";
    times[4] = simdTime.total!"usecs";

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
        k[bitReverse(g[i], P)] = i;
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
