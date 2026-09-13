#include <metal_stdlib>
using namespace metal;

inline float read_value(VALUE value) { return READ_VALUE; }
inline VALUE write_value(float value) { return WRITE_VALUE; }

// One thread owns a complete output row, including its LSE. This also makes
// in-place pairwise merging safe when out or out_lse aliases a left input.
kernel void merge_pair(
    device VALUE* output, device const VALUE* left, device const float* llse,
    device const VALUE* right, device const float* rlse, device float* olse,
    constant long* p, uint gid [[thread_position_in_grid]]) {
    long t=gid/p[1], h=gid%p[1];
    float a=llse[h*p[12]+t*p[13]], b=rlse[h*p[14]+t*p[15]];
    float maximum=max(a,b);
    bool empty=maximum == -INFINITY;
    float wa=empty ? 0.0f : (maximum == INFINITY ? float(a==maximum) : exp(a-maximum));
    float wb=empty ? 0.0f : (maximum == INFINITY ? float(b==maximum) : exp(b-maximum));
    float total=wa+wb;
    for (long d=0; d<p[2]; ++d) {
        float value=0.0f;
        if (wa>0.0f) value+=wa*read_value(left[t*p[6]+h*p[7]+d*p[8]]);
        if (wb>0.0f) value+=wb*read_value(right[t*p[9]+h*p[10]+d*p[11]]);
        if (!empty) value/=total;
        output[t*p[3]+h*p[4]+d*p[5]]=write_value(value);
    }
    if (p[18]) olse[h*p[16]+t*p[17]]=empty ? -INFINITY : maximum+log(total);
}

kernel void merge_partials(
    device const VALUE* partial, device const float* lse,
    device VALUE* output, device float* output_lse,
    constant long* p, uint gid [[thread_position_in_grid]]) {
    long t=gid/p[2], h=gid%p[2];
    float maximum=-INFINITY;
    for (long s=0; s<p[0]; ++s) maximum=max(maximum,lse[s*p[8]+h*p[9]+t*p[10]]);
    bool empty=maximum == -INFINITY;
    float denominator=0.0f;
    if (!empty) {
        for (long s=0; s<p[0]; ++s) {
            float l=lse[s*p[8]+h*p[9]+t*p[10]];
            denominator+=maximum==INFINITY ? float(l==maximum) : exp(l-maximum);
        }
    }
    for (long d=0; d<p[3]; ++d) {
        float value=0.0f;
        if (!empty) {
            for (long s=0; s<p[0]; ++s) {
                float l=lse[s*p[8]+h*p[9]+t*p[10]];
                float weight=maximum==INFINITY ? float(l==maximum) : exp(l-maximum);
                if (weight>0.0f) value+=weight*read_value(partial[s*p[4]+t*p[5]+h*p[6]+d*p[7]]);
            }
            value/=denominator;
        }
        output[(t*p[2]+h)*p[3]+d]=write_value(value);
    }
    output_lse[h*p[1]+t]=empty ? -INFINITY : maximum+log(denominator);
}
