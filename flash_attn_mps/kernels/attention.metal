// Native forward attention. The SIMD matrix fragments prepended to this source
// are from HF/MLX STEEL; all indexing below operates on PyTorch tensor strides.

inline float fp8_e4m3(uchar bits) {
  uint exponent = (bits >> 3) & 15, mantissa = bits & 7;
  float x = exponent == 0 ? float(mantissa) * 0.001953125f
                         : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
  if (exponent == 15 && mantissa == 7) x = NAN;
  return (bits & 128) ? -x : x;
}
template<typename T> inline float read_element(const device T* p, long offset) {
  return float(p[offset]);
}
template<> inline float read_element<uchar>(const device uchar* p, long offset) {
  return fp8_e4m3(p[offset]);
}

inline long cache_offset(constant long* p, const device int* table, int b,
                         int token, int head, bool value) {
  int s = value ? VS0 : KS0;
  if (p[PAGED]) {
    int page = int(p[PAGE]);
    int block = table[b * p[BT0] + (token / page) * p[BT1]];
    return block * p[s] + (token % page) * p[s+1] + head * p[s+2];
  }
  return token * p[s+1] + head * p[s+2];
}

inline bool allowed_key(constant long* p, int b, int token, int q_abs,
                        int packed_q, bool causal, const device int* ranges,
                        const device int* prefix, const device int* window) {
  int diff = q_abs - token;
  if (p[MASK] == 1) {
    bool local = diff >= 0 && (p[MASK_WIN] < 0 || diff < p[MASK_WIN]);
    int lo = ranges[packed_q * p[MR0]], hi = ranges[packed_q * p[MR0] + p[MR1]];
    bool modality = lo >= 0 && token >= lo && token <= hi;
    if (p[CLAMP]) modality = modality && (p[MASK_WIN] < 0 || diff < p[MASK_WIN]);
    return local || modality;
  }
  if (p[MASK] == 2) {
    return diff >= 0 && (token < prefix[b * p[MP0]] || diff < window[0]);
  }
  return (!causal || diff >= 0) && (p[WIN_L] < 0 || diff <= p[WIN_L])
      && (p[WIN_R] < 0 || -diff <= p[WIN_R]);
}

inline float attention_score(float dot, float scale, float cap, float slope, int diff) {
  float score = dot * scale;
  if (cap > 0) score = cap * precise::tanh(score / cap);
  return score - slope * float(abs(diff));
}

struct AttnMax { static float apply(float x, float y) { return max(x,y); } };
struct AttnSum { static float apply(float x, float y) { return x+y; } };
struct AttnExp { static float apply(float x, float y) { return isfinite(y) ? fast::exp2(x-y) : 0.0f; } };
struct AttnMul { static float apply(float x, float y) { return x*y; } };

// One query tile per (sequence, head, split). K/V remain paged on device;
// only a bounded tile is staged in threadgroup memory at any time.
kernel void attention_tiled(
    const device QType* Q [[buffer(0)]], const device KType* K [[buffer(1)]],
    const device VType* V [[buffer(2)]], device OType* O [[buffer(3)]],
    device float* LSE [[buffer(4)]], constant long* p [[buffer(5)]],
    constant float* f [[buffer(6)]], const device int* cuq [[buffer(7)]],
    const device int* cuk [[buffer(8)]], const device int* used [[buffer(9)]],
    const device int* table [[buffer(10)]], const device bool* causal_by_batch [[buffer(11)]],
    const device float* alibi [[buffer(12)]], const device float* sinks [[buffer(13)]],
    const device float* qs [[buffer(14)]], const device float* ks [[buffer(15)]],
    const device float* vs [[buffer(16)]], const device int* ranges [[buffer(17)]],
    const device int* prefix [[buffer(18)]], const device int* window [[buffer(19)]],
    const device int* leftpad [[buffer(20)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  int b = group.z / p[SPLITS], split = group.z % p[SPLITS];
  int h = group.y, kh = h / (p[HQ] / p[HK]);
  int q_begin = cuq[b * p[CQ0]], q_len = cuq[(b+1)*p[CQ0]] - q_begin;
  int q_block = group.x * BQ;
  if (q_block >= q_len) return;
  int k_cu_begin = p[CUK] ? cuk[b*p[CK0]] : 0;
  int k_begin = p[PAGED] ? 0 : k_cu_begin;
  int k_len = p[USED] ? used[b*p[U0]] : cuk[(b+1)*p[CK0]] - k_cu_begin;
  int left = p[LEFTPAD] ? leftpad[b*p[LP0]] : 0;
  bool causal = p[DYNAMIC] ? causal_by_batch[b*p[CA0]] : bool(p[CAUSAL]);
  float scale = f[0] * (p[QDS] ? qs[b*p[QD0]+kh*p[QD1]] : 1.0f)
                     * (p[KDS] ? ks[b*p[KD0]+kh*p[KD1]] : 1.0f);
  float vscale = p[VDS] ? vs[b*p[VD0]+kh*p[VD1]] : 1.0f;
  float slope = p[ALIBI] ? alibi[b*p[A0]+h*p[A1]] : 0.0f;
  float sink = p[SINK] && split == 0 ? sinks[h*p[SI0]] : -INFINITY;
  constexpr int PAD=16/sizeof(SharedType), LQ = BD+PAD, LK = BK+PAD, LV = BV+PAD;
  constexpr int KV_SIZE = BD*LK > BK*LV ? BD*LK : BK*LV;
  threadgroup SharedType qmem[BQ*LQ], kvmem[KV_SIZE];
  int tid = sg*32+lane;
  for (int z=tid; z<BQ*BD; z+=WM*32) {
    int r=z/BD, d=z%BD;
    qmem[r*LQ+d] = SharedType(r+q_block<q_len && d<p[DQ]
      ? read_element(Q,(q_begin+q_block+r)*p[QS0]+h*p[QS1]+d*p[QS2]) : 0.0f);
  }
  using Frag = BaseMMAFrag<float,8,8>;
  constexpr int TK=BK/8, TV=BV/8;
  MMATile<float,1,1,Frag> qt, vt;
  MMATile<float,1,TK,Frag> kt, scores;
  MMATile<float,1,TV,Frag> result;
  result.clear();
  short2 coord=Frag::get_coord(lane);
  int sm=coord.y, sn=coord.x, row=sg*8+sm;
  float max_score[1]={sink*1.4426950408889634f};
  float sum_score[1]={isfinite(sink) ? 1.0f : 0.0f};
  int tile_count=(k_len+BK-1)/BK;
  int first=(tile_count*split)/p[SPLITS], last=(tile_count*(split+1))/p[SPLITS];
  // Skip whole tiles beyond a causal query tile when no mask override is active.
  if (causal && p[MASK]==0)
    last=min(last,max(0,(min(q_block+BQ,q_len)+k_len-q_len+BK-1)/BK));
  for (int kb=first; kb<last; ++kb) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int z=tid; z<BK*BD; z+=WM*32) {
      int r=z/BD,d=z%BD,t=kb*BK+r;
      long offset=0;
      if (t<k_len && d<p[DQ]) offset=cache_offset(p,table,b,k_begin+left+t,kh,false)+d*p[KS3];
      kvmem[d*LK+r]=SharedType(t<k_len && d<p[DQ] ? read_element(K,offset) : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scores.clear();
    STEEL_PRAGMA_UNROLL
    for (int d=0; d<BD/8; ++d) {
      qt.template load<SharedType,1,1,LQ,1>(&qmem[row*LQ+sn+d*8]);
      kt.template load<SharedType,1,1,LK,1>(&kvmem[sm*LK+sn+d*8*LK]);
      tile_matmad(scores,qt,kt,scores);
    }
    int q_abs=q_block+row+k_len-q_len;
    STEEL_PRAGMA_UNROLL
    for (int j=0;j<TK;++j) for(int e=0;e<2;++e) {
      int t=kb*BK+sn+j*8+e;
      bool keep=t<k_len && q_block+row<q_len
        && allowed_key(p,b,t,q_abs,q_begin+q_block+row,causal,ranges,prefix,window);
      scores.frag_at(0,j)[e]=keep
        ? attention_score(scores.frag_at(0,j)[e],scale,f[1],slope,q_abs-t)*1.4426950408889634f
        : -INFINITY;
    }
    float next_max[1]={max_score[0]};
    scores.template row_reduce<AttnMax>(next_max);
    scores.template row_bin_op<AttnExp>(next_max);
    float factor[1]={isfinite(next_max[0]) ? fast::exp2(max_score[0]-next_max[0]) : 0.0f};
    sum_score[0]*=factor[0];
    scores.template row_reduce<AttnSum>(sum_score);
    result.template row_bin_op<AttnMul>(factor);
    max_score[0]=next_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int z=tid;z<BK*BV;z+=WM*32) {
      int r=z/BV,d=z%BV,t=kb*BK+r;
      long offset=0;
      if(t<k_len && d<p[DV]) offset=cache_offset(p,table,b,k_begin+left+t,kh,true)+d*p[VS3];
      kvmem[r*LV+d]=SharedType(t<k_len && d<p[DV] ? read_element(V,offset) : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for(int j=0;j<TV;++j) {
      STEEL_PRAGMA_UNROLL
      for(int k=0;k<TK;++k) {
      vt.template load<SharedType,1,1,LV,1>(&kvmem[(sm+k*8)*LV+sn+j*8]);
      Frag::mma(result.frag_at(0,j),scores.frag_at(0,k),vt.frag_at(0,0),result.frag_at(0,j));
      }
    }
  }
  if(q_block+row<q_len) {
    int t=q_begin+q_block+row;
    if(sn==0) LSE[split*p[T]*p[HQ]+h*p[T]+t]=sum_score[0]>0
      ? max_score[0]*0.6931471805599453f+log(sum_score[0]) : -INFINITY;
    float norm=sum_score[0]>0 ? vscale/sum_score[0] : 0.0f;
    for(int j=0;j<TV;++j) for(int e=0;e<2;++e) {
      int d=sn+j*8+e;
      if(d<p[DV]) O[split*p[OSPLIT]+t*p[OS0]+h*p[OS1]+d*p[OS2]]
        =OType(result.frag_at(0,j)[e]*norm);
    }
  }
}

// SIMD-vector online softmax is used for single-query decode and uncommon
// dimensions. A split owns its complete reduction, so no atomic writes occur.
kernel void attention_vector(
    const device QType* Q [[buffer(0)]], const device KType* K [[buffer(1)]],
    const device VType* V [[buffer(2)]], device OType* O [[buffer(3)]],
    device float* LSE [[buffer(4)]], constant long* p [[buffer(5)]],
    constant float* f [[buffer(6)]], const device int* cuq [[buffer(7)]],
    const device int* cuk [[buffer(8)]], const device int* used [[buffer(9)]],
    const device int* table [[buffer(10)]], const device bool* causal_by_batch [[buffer(11)]],
    const device float* alibi [[buffer(12)]], const device float* sinks [[buffer(13)]],
    const device float* qs [[buffer(14)]], const device float* ks [[buffer(15)]],
    const device float* vs [[buffer(16)]], const device int* ranges [[buffer(17)]],
    const device int* prefix [[buffer(18)]], const device int* window [[buffer(19)]],
    const device int* leftpad [[buffer(20)]],
    uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  int b=group.z/p[SPLITS],split=group.z%p[SPLITS],h=group.y,qi=group.x;
  int start=cuq[b*p[CQ0]],qlen=cuq[(b+1)*p[CQ0]]-start;
  if(qi>=qlen)return;
  int kh=h/(p[HQ]/p[HK]),tq=start+qi;
  int kcu=p[CUK]?cuk[b*p[CK0]]:0;
  int kb=p[PAGED]?0:kcu;
  int len=p[USED]?used[b*p[U0]]:cuk[(b+1)*p[CK0]]-kcu;
  int left=p[LEFTPAD]?leftpad[b*p[LP0]]:0;
  int q_abs=qi+len-qlen;
  bool causal=p[DYNAMIC]?causal_by_batch[b*p[CA0]]:bool(p[CAUSAL]);
  float scale=f[0]*(p[QDS]?qs[b*p[QD0]+kh*p[QD1]]:1.0f)
                  *(p[KDS]?ks[b*p[KD0]+kh*p[KD1]]:1.0f);
  float vscale=p[VDS]?vs[b*p[VD0]+kh*p[VD1]]:1.0f;
  float slope=p[ALIBI]?alibi[b*p[A0]+h*p[A1]]:0.0f;
  float m=p[SINK]&&split==0?sinks[h*p[SI0]]:-INFINITY;
  float denominator=isfinite(m)?1.0f:0.0f;
  float query[(BD+31)/32],acc[(BV+31)/32];
  for(int j=0;j<(BD+31)/32;++j) {
    int d=lane+j*32;
    query[j]=d<p[DQ]?read_element(Q,tq*p[QS0]+h*p[QS1]+d*p[QS2]):0;
  }
  for(int j=0;j<(BV+31)/32;++j)acc[j]=0;
  int begin=(len*split)/p[SPLITS],end=(len*(split+1))/p[SPLITS];
  if(causal && p[MASK]==0)end=min(end,q_abs+1);
  if(p[MASK]==0 && p[WIN_L]>=0)begin=max(begin,q_abs-int(p[WIN_L]));
  for(int t=begin;t<end;++t) {
    if(!allowed_key(p,b,t,q_abs,tq,causal,ranges,prefix,window))continue;
    long offset=cache_offset(p,table,b,kb+left+t,kh,false);
    float dot=0;
    for(int j=0;j<(BD+31)/32;++j) {
      int d=lane+j*32;
      if(d<p[DQ])dot=fma(query[j],read_element(K,offset+d*p[KS3]),dot);
    }
    dot=simd_sum(dot);
    float score=attention_score(dot,scale,f[1],slope,q_abs-t);
    float next=max(m,score),old_weight=fast::exp(m-next),weight=fast::exp(score-next);
    denominator=denominator*old_weight+weight;
    m=next;
    long vo=cache_offset(p,table,b,kb+left+t,kh,true);
    for(int j=0;j<(BV+31)/32;++j) {
      int d=lane+j*32;
      if(d<p[DV])acc[j]=fma(weight,read_element(V,vo+d*p[VS3]),acc[j]*old_weight);
    }
  }
  float norm=denominator>0?vscale/denominator:0;
  for(int j=0;j<(BV+31)/32;++j) {
    int d=lane+j*32;
    if(d<p[DV])O[split*p[OSPLIT]+tq*p[OS0]+h*p[OS1]+d*p[OS2]]=OType(acc[j]*norm);
  }
  if(lane==0)LSE[split*p[T]*p[HQ]+h*p[T]+tq]=denominator>0?m+log(denominator):-INFINITY;
}
