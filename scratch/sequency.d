// extern (C) void run_FWT(const int Pa, const int Na, const int N,
        // float *fi, float *Fa, float *seq, const int blockDimX,
        // const int gridDimX);

import std.stdio;

// import cuda_d.cuda;
// import cuda_d.cuda_runtime_api;
// import cuda_d.cublas_api;

void main()
{
    int P = 3;
    int N = 2^^P;

    int[] s;
    s.length = N;
    for(int i=0;i<N;i++) {
        s[i] = i;
    }
    auto k = getSequence(s, N, P);
    writeln("k\n", k);

    int[] krev;
    krev.length = k.length;
    for(int i=0;i<N;i++) {
        krev[k[i]] = i;
    }
    writeln("krev\n", krev);

    int i = 1;
    foreach (j; 0 .. 10) {
        writeln(j, ", ", i);
        i <<= 1;
    }
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
