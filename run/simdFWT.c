
#include <stdio.h>
#include <immintrin.h>

void getSequence(int *s, int Na, int P, int *seq);
int bitReverse(int num, int numBits);

void run_FWT_SIMD(const int Pa, const int Na, const int N,
        const float *fi, float *Fa, const int *seq);

void simd_FWT(float *fi, float *Fa, int *seq, const int Pa, const int Na);
void view8floats(__m256 a);
void view8ints(__m256i a);

/*
int main()
{
    const int Pa = 3;
    const int Na = 2<<(Pa-1);

    float fi[Na];
    for (int i=0; i<Na; ++i) {
        fi[i] = 1.0 * i;
    }

    float Fa[Na];

    // calculate sequency mapping (same for each FWT)
    int s[Na];
    for(int i=0;i<Na;++i) {
        s[i] = i;
    }
    int seq[Na];
    getSequence(s, Na, Pa, seq);

    simd_FWT(fi, Fa, seq, Pa, Na);
}
*/

void run_FWT_SIMD(const int Pa, const int Na, const int N,
        const float *fi, float *Fa, const int *seq)
{
    for (int i=0; i<N; i+=Na) {
        simd_FWT(fi+i, Fa+i, seq+i, Pa, Na);
    }
}

void simd_FWT(float *fi, float *Fa, int *seqArr, const int Pa, const int Na)
{
    // __m256 F1[4];
    // __m256 F2[4];

    // __m256i seq[4];
    // __m256i srcMask[4];
    // __m256i negMask[4];

    // for (int i=0; i<Na; i+=8) {
        // F1[i] = __m256_set_ps(
                // fi[0+i], fi[1+i], fi[2+i], fi[3+i],
                // fi[4+i], fi[5+i], fi[6+i], fi[7+i]);
    // }

    // -----

    __m256 F1;
    __m256 F2;

    __m256i lid;
    __m256i seq;
    __m256i srcMask;
    __m256i negMask;

    __m256i a;
    __m256i one;
    __m256i two;
    __m256i negOne;
    __m256i Nm;

    one = _mm256_set1_epi32(1);
    two = _mm256_set1_epi32(2);
    negOne = _mm256_set1_epi32(-1);
    Nm = _mm256_set1_epi32(Na/2);

    // setr because Intel has it flipped
    F1 = _mm256_setr_ps(fi[0], fi[1], fi[2], fi[3], fi[4], fi[5], fi[6],
            fi[7]);

    // setr because Intel has it flipped
    seq = _mm256_setr_epi32(seqArr[0], seqArr[1], seqArr[2], seqArr[3],
            seqArr[4], seqArr[5], seqArr[6], seqArr[7]);

    // setr because Intel has it flipped
    lid = _mm256_setr_epi32(0,1,2,3,4,5,6,7);

    for(int pm=0;pm<Pa;pm++) {

        a = _mm256_set1_epi32(Pa-pm-1);

        // -- srcMask --
        // bitwise shift right by 'a'
        srcMask = _mm256_srlv_epi32(lid, a);
        // bitwise and
        srcMask = _mm256_castps_si256(
            _mm256_and_ps(_mm256_castsi256_ps(srcMask), _mm256_castsi256_ps(one)));
        // bitwise xor
        srcMask = _mm256_castps_si256(
            _mm256_xor_ps(_mm256_castsi256_ps(srcMask), _mm256_castsi256_ps(one)));


        // -- negMask --
        // bitwise shift right by 'a'
        negMask = _mm256_srlv_epi32(lid, a);
        // bitwise and
        negMask = _mm256_castps_si256(
                _mm256_and_ps(_mm256_castsi256_ps(negMask), _mm256_castsi256_ps(one)));
        // mult by 2
        negMask = _mm256_mullo_epi32(negMask, two);
        // take 1
        negMask = _mm256_sub_epi32(negMask, one);
        // mult by neg 1
        negMask = _mm256_mullo_epi32(negMask, negOne);


        // -- shuffle down --
        F2 = _mm256_permutevar8x32_ps(F1, _mm256_add_epi32(lid, Nm));
        F2 = _mm256_mul_ps(_mm256_cvtepi32_ps(srcMask), F2);

        
        // -- flip mask --
        srcMask = _mm256_castps_si256(
            _mm256_xor_ps(_mm256_castsi256_ps(srcMask), _mm256_castsi256_ps(one)));


        // -- shuffle up --
        F2 = _mm256_add_ps(F2,
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(srcMask),
                        _mm256_permutevar8x32_ps(F1,
                            _mm256_sub_epi32(lid, Nm))));


        // -- add to existing value using negMask --
        F1 = _mm256_add_ps(
                _mm256_mul_ps(F1,
                    _mm256_cvtepi32_ps(negMask)), F2);


        // -- update shfl width --
        // right shift by 1
        Nm = _mm256_srlv_epi32(Nm, one);
    }

    __attribute__ ((aligned (32))) float output[8];
    _mm256_store_ps(output, F1);

    printf(""); // WEIRD! core dumps unless something is here.. DEBUG

    for (int i=0; i<Na; ++i) {
        Fa[i] = output[i];
    }
}

void getSequence(int *s, int N, int P, int *seq)
{
    int *g;
    g = (int *)malloc(N * sizeof(int));
    seq = (int *)malloc(N * sizeof(int));

    for (int i=0;i<N;i++) {
        g[i] = s[i] ^ (s[i] >> 1);
    }

    for (int i=0;i<N;i++) {
        seq[bitReverse(g[i], P)] = i;
    }
}

int bitReverse(int num, int numBits)
{ 
    int reverse_num = 0; 
    for (int i = 0; i < numBits; i++) 
    { 
        if((num & (1 << i))) 
           reverse_num |= 1 << ((numBits - 1) - i);   
   } 
    return reverse_num; 
} 


void view8floats(__m256 a)
{
    __attribute__ ((aligned (32))) float output[8];
    _mm256_store_ps(output, a);
    printf("%f %f %f %f %f %f %f %f\n",
         output[0], output[1], output[2], output[3],
         output[4], output[5], output[6], output[7]);
}

void view8ints(__m256i a)
{
    __attribute__ ((aligned (32))) int output[8];
    // _mm256_store_epi32(output, a); // only avx512..
    _mm256_store_si256(output, a);
    printf("%d %d %d %d %d %d %d %d\n",
         output[0], output[1], output[2], output[3],
         output[4], output[5], output[6], output[7]);
}
