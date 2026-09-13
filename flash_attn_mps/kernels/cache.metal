#include <metal_stdlib>
using namespace metal;

inline float read_value(SRC value) { return READ_VALUE; }
inline SRC write_value(float value) { return WRITE_VALUE; }

// E4M3FN reserves 0x7f for NaN; the largest finite value is 448.
inline uchar encode_e4m3(float value) {
    uint sign = (as_type<uint>(value) >> 24) & 128;
    float a = abs(value);
    if (isnan(a)) return uchar(sign | 127);
    if (a >= 448.0f) return uchar(sign | 126);
    if (a < 0.015625f) return uchar(sign | uint(rint(a * 512.0f)));
    int exponent = int(floor(log2(a)));
    int mantissa = int(rint(ldexp(a, 3 - exponent))) - 8;
    int code = ((exponent + 7) << 3) + mantissa;
    return uchar(sign | uint(min(code, 126)));
}

kernel void store(
    device const SRC* input, device DST* cache, device const INDEX* slots,
    device const float* scale, constant long* p,
    uint3 gid [[thread_position_in_grid]]) {
    uint d=gid.x, h=gid.y, t=gid.z;
    long slot = long(slots[t * p[10]]);
    if (slot < 0 || slot >= p[11]) return;
    long block, offset;
    if (p[11]<=0xffffffffL) {
        uint token=uint(slot), page_size=uint(p[0]);
        block=token/page_size;
        offset=token%page_size;
    } else {
        block=slot/p[0];
        offset=slot%p[0];
    }
    long src = t*p[3] + h*p[4] + d*p[5];
    long dst = block*p[6] + offset*p[7] + h*p[8] + d*p[9];
    cache[dst] = STORE_VALUE;
}

kernel void append(
    device const SRC* input, device DST* cache, device const int* lengths,
    device const int* batch_index,
    device const int* table, device const float* scale, constant long* p,
    uint3 gid [[thread_position_in_grid]]) {
    uint d=gid.x, h=gid.y, j=gid.z%uint(p[1]), b=gid.z/uint(p[1]);
    long row=p[18] ? long(batch_index[b]) : b, pos=long(lengths[b])+j;
    if (row<0 || row>=p[17] || pos<0 || pos>=(p[14] ? p[15]*p[0] : p[0])) return;
    uint token=uint(pos), page_size=uint(p[0]);
    long block=p[14] ? long(table[row*p[15]+token/page_size]) : row;
    long offset=p[14] ? token%page_size : pos;
    if (block<0 || block>=p[16] || offset<0 || offset>=p[0]) return;
    long src=b*p[4]+j*p[5]+h*p[6]+d*p[7];
    long dst=block*p[8]+offset*p[9]+h*p[10]+d*p[11];
    cache[dst] = APPEND_VALUE;
}

kernel void rotate(
    device const SRC* input, device SRC* output,
    device const float* cosine, device const float* sine,
    device const int* lengths, constant long* p,
    uint3 gid [[thread_position_in_grid]]) {
    uint d=gid.x, h=gid.y, j=gid.z%uint(p[1]), b=gid.z/uint(p[1]);
    long src=b*p[4]+j*p[5]+h*p[6]+d*p[7];
    long dst=(long(gid.z)*p[2]+h)*p[3]+d;
    if (d>=p[8]*2) { output[dst]=input[src]; return; }
    uint pair=p[9] ? d/2 : d%uint(p[8]);
    bool second=p[9] ? bool(d%2) : d>=p[8];
    long other=p[9] ? (second ? d-1 : d+1) : (second ? d-p[8] : d+p[8]);
    long other_src=b*p[4]+j*p[5]+h*p[6]+other*p[7];
    long pos=long(lengths[b])+(p[10] ? j : 0);
    float c=cosine[pos*p[8]+pair], s=sine[pos*p[8]+pair];
    float x=read_value(input[src]), y=read_value(input[other_src]);
    float value=x*c+(second ? y*s : -y*s);
    output[dst]=write_value(value);
}
