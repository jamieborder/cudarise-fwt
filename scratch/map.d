import std.stdio;
import std.algorithm.iteration: map, fold;


void main()
{
    const int N = 8;
    float[] diff;
    diff.length = N;
    foreach (i; 0 .. N) {
        diff[i] = i;
    }

    writeln("L2 error for GPU (shfls): ", fold!((a, b) => a + b)(map!(a => a^^2.0)(diff)));
}
