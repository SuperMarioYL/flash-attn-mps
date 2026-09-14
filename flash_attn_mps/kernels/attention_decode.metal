// Segmented SIMD-vector decoding, based on the reduction strategy in
// PyTorch v2.14.0 aten/src/ATen/native/mps/kernels/DecodeAttention.h (BSD-3-Clause),
// itself adapted from Apple MLX (MIT). See the distributed third-party notices.
// The paged/strided addressing, grouped-query reuse, FP32 partial states and
// direct-output merge below are additions for flash-attn-mps.
#include <metal_stdlib>
using namespace metal;

template<typename T> inline float read_decode(T value) { return float(value); }
template<> inline float read_decode<uchar>(uchar bits) {
  uint exponent=(bits>>3)&15,mantissa=bits&7;
  float value=exponent==0?float(mantissa)*0.001953125f
    :ldexp(1.0f+float(mantissa)*0.125f,int(exponent)-7);
  if(exponent==15 && mantissa==7)value=NAN;
  return bits&128?-value:value;
}

kernel void decode_partitions(
    const device Query* query [[buffer(0)]],
    const device Key* key [[buffer(1)]],
    const device Value* value [[buffer(2)]],
    device Partial* output [[buffer(3)]],
    device float* lse [[buffer(4)]],
    constant long* p [[buffer(5)]], constant float* math_params [[buffer(6)]],
    const device int* cuq [[buffer(7)]], const device int* cuk [[buffer(8)]],
    const device int* lengths [[buffer(9)]], const device int* table [[buffer(10)]],
    const device int* leftpad [[buffer(11)]],
    const device float* qs [[buffer(12)]],const device float* ks [[buffer(13)]],
    const device float* vs [[buffer(14)]],const device float* sinks [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int QPT=(DQ+31)/32, VPT=(DV+31)/32, PAD=SG_COUNT+1;
  threadgroup float maxima[Q_GROUP*SG_COUNT], sums[Q_GROUP*SG_COUNT];
  threadgroup float states[Q_GROUP*VPT*32*PAD];
  int b=group.y, part=group.z, h0=int(group.x)*Q_GROUP,splits=int(p[PARTS]);
  int token=cuq[b*p[CQ]], qend=cuq[(b+1)*p[CQ]];
  if(token==qend)return;
  int kh=h0/int(p[GQA]);
  int ck=HAS_CUK?cuk[b*p[CK]]:0;
  int len=HAS_USED?lengths[b*p[USED]]:cuk[(b+1)*p[CK]]-ck;
  int left=HAS_LEFT?leftpad[b*p[LEFT]]:0;
  int visible_begin=HAS_WINDOW?max(0,len-int(p[WINDOW_LEFT])-1):0;
  int begin=visible_begin+part*int(p[SPAN]),end=min(len,begin+int(p[SPAN]));
  if(!PAGED)left+=ck;
  float scale=math_params[0]*(HAS_Q_SCALE?qs[b*p[QSC0]+kh*p[QSC1]]:1.0f)
                             *(HAS_K_SCALE?ks[b*p[KSC0]+kh*p[KSC1]]:1.0f);
  float vscale=HAS_V_SCALE?vs[b*p[VSC0]+kh*p[VSC1]]:1.0f;
  float q[Q_GROUP][QPT], acc[Q_GROUP][VPT];
  float m[Q_GROUP], z[Q_GROUP];
  #pragma unroll
  for(int g=0;g<Q_GROUP;++g) {
    #pragma unroll
    for(int j=0;j<QPT;++j) {
      int d=int(lane)*QPT+j;
      q[g][j]=d<DQ?read_decode(query[token*p[Q0]+(h0+g)*p[Q1]+d*(UNIT_D?1:p[Q2])])*scale:0.0f;
    }
    #pragma unroll
    for(int j=0;j<VPT;++j)acc[g][j]=0;
    bool seed=HAS_SINK && part==0 && sg==0;
    m[g]=seed?sinks[(h0+g)*p[SINK0]]:-INFINITY;z[g]=seed?1.0f:0.0f;
  }
  const device Key* kpage=key;
  const device Value* vpage=value;
  int old_page=-1;
  for(int t=begin+int(sg);t<end;t+=SG_COUNT) {
    int logical=t+left;
    long ko,vo;
    if(PAGED) {
      int page=logical/PAGE_SIZE;
      if(page!=old_page) {
        int physical=table[b*p[BT0]+page*p[BT1]];
        kpage=key+physical*p[K0]+kh*p[K2];
        vpage=value+physical*p[V0]+kh*p[V2];
        old_page=page;
      }
      ko=(logical%PAGE_SIZE)*p[K1];
      vo=(logical%PAGE_SIZE)*p[V1];
    } else {
      ko=logical*p[K1]+kh*p[K2];
      vo=logical*p[V1]+kh*p[V2];
    }
    float kv[QPT],vv[VPT];
    #pragma unroll
    for(int j=0;j<QPT;++j) {
      int d=int(lane)*QPT+j;
      kv[j]=d<DQ?read_decode(kpage[ko+d*(UNIT_D?1:p[K3])]):0.0f;
    }
    #pragma unroll
    for(int j=0;j<VPT;++j) {
      int d=int(lane)*VPT+j;
      vv[j]=d<DV?read_decode(vpage[vo+d*(UNIT_D?1:p[V3])]):0.0f;
    }
    #pragma unroll
    for(int g=0;g<Q_GROUP;++g) {
      float score=0;
      #pragma unroll
      for(int j=0;j<QPT;++j)score=fma(q[g][j],kv[j],score);
      score=simd_sum(score);
      if(HAS_CAP)score=math_params[1]*precise::tanh(score/math_params[1]);
      float next=max(m[g],score);
      float factor=fast::exp(m[g]-next),weight=fast::exp(score-next);
      z[g]=fma(z[g],factor,weight);
      #pragma unroll
      for(int j=0;j<VPT;++j)acc[g][j]=fma(vv[j],weight,acc[g][j]*factor);
      m[g]=next;
    }
  }
  #pragma unroll
  for(int g=0;g<Q_GROUP;++g)if(lane==0) {
    maxima[g*SG_COUNT+sg]=m[g];sums[g*SG_COUNT+sg]=z[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total[Q_GROUP],peak[Q_GROUP];
  #pragma unroll
  for(int g=0;g<Q_GROUP;++g) {
    float lane_max=lane<SG_COUNT?maxima[g*SG_COUNT+lane]:-INFINITY;
    peak[g]=simd_max(lane_max);
    float lane_weight=isfinite(peak[g])?fast::exp(lane_max-peak[g]):0;
    total[g]=simd_sum(lane<SG_COUNT?sums[g*SG_COUNT+lane]*lane_weight:0.0f);
    float factor=isfinite(peak[g])?fast::exp(m[g]-peak[g]):0;
    #pragma unroll
    for(int j=0;j<VPT;++j)
      states[((g*VPT+j)*32+lane)*PAD+sg]=acc[g][j]*factor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) {
    #pragma unroll
    for(int g=0;g<Q_GROUP;++g) {
      int h=h0+g;
      float inverse=total[g]>0?vscale/total[g]:0;
      #pragma unroll
      for(int j=0;j<VPT;++j) {
        int d=int(lane)*VPT+j;
        float sum=0;
        #pragma unroll
        for(int r=0;r<SG_COUNT;++r)sum+=states[((g*VPT+j)*32+lane)*PAD+r];
        if(d<DV) {
          long destination=PARTITIONED?((token*p[HQ]+h)*splits+part)*DV+d
            :token*p[O0]+h*p[O1]+d*p[O2];
          output[destination]=Partial(sum*inverse);
        }
      }
      if(lane==0) {
        long destination=PARTITIONED?(token*p[HQ]+h)*splits+part:h*p[T]+token;
        lse[destination]=total[g]>0?peak[g]+log(total[g]):-INFINITY;
      }
    }
  }
}

kernel void decode_merge(
    const device float* partials [[buffer(0)]], const device float* partial_lse [[buffer(1)]],
    device Out* output [[buffer(2)]], device float* lse [[buffer(3)]],
    constant long* p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER_LANE=(DV+31)/32;
  threadgroup float weights[32];
  int splits=int(p[PARTS]);
  float peak=-INFINITY;
  long row_base=long(row)*splits;
  for(int s=lane;s<splits;s+=32)peak=max(peak,partial_lse[row_base+s]);
  peak=simd_max(peak);
  float sum=0,first_weight=0;
  for(int s=lane;s<splits;s+=32) {
    float w=isfinite(peak)?fast::exp(partial_lse[row_base+s]-peak):0;
    if(s==int(lane))first_weight=w;
    sum+=w;
  }
  sum=simd_sum(sum);
  float accumulator[PER_LANE];
  #pragma unroll
  for(int j=0;j<PER_LANE;++j)accumulator[j]=0;
  // Fixed shared memory and a runtime chunk loop prevent recompilation as KV
  // length grows or the caller changes the requested split count.
  for(int chunk=0;chunk<splits;chunk+=32) {
    int count=min(32,splits-chunk);
    weights[lane]=chunk==0?first_weight:
      (int(lane)<count && isfinite(peak)?fast::exp(partial_lse[row_base+chunk+lane]-peak):0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(count==32) {
      #pragma unroll
      for(int s=0;s<32;++s) {
        #pragma unroll
        for(int j=0;j<PER_LANE;++j) {
          int d=int(lane)+j*32;
          if(d<DV)accumulator[j]=fma(weights[s],partials[(row_base+chunk+s)*DV+d],accumulator[j]);
        }
      }
    } else {
      for(int s=0;s<count;++s) {
        #pragma unroll
        for(int j=0;j<PER_LANE;++j) {
          int d=int(lane)+j*32;
          if(d<DV)accumulator[j]=fma(weights[s],partials[(row_base+chunk+s)*DV+d],accumulator[j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  int token=row/int(p[HQ]),h=row%int(p[HQ]);
  #pragma unroll
  for(int j=0;j<PER_LANE;++j) {
    int d=int(lane)+j*32;
    if(d<DV)output[token*p[O0]+h*p[O1]+d*p[O2]]=Out(sum>0?accumulator[j]/sum:0);
  }
  if(lane==0)lse[h*p[T]+token]=sum>0?peak+log(sum):-INFINITY;
}
