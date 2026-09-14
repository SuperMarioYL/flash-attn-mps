// HF/MLX tiled path. Provenance and licenses: NOTICE.
struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale
  float softcapping; ///< Softcapping value (0 disables the compile-time branch)

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  int64_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  int64_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
  
  // Flash Attention variable-length support
  int total_q_tokens; ///< Total number of query tokens (sum of all sequence lengths)
  int total_k_tokens; ///< Total number of key/value tokens
  int max_seqlen_q; ///< Maximum query sequence length
  int max_seqlen_k; ///< Maximum key/value sequence length
};

struct AttnMaskParams {
  int64_t M_strides[3]; ///< Mask  strides (B, H, qL, kL = 1)
};

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q = false;
constant bool align_K = false;

constant bool has_mask = false;
constant bool do_causal = __CAUSAL__;
constant bool do_softcap = __SOFTCAP__;

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return isfinite(y) ? fast::exp2(x - y) : 0.0f;
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return y > 0 ? x / y : 0.0f;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    constant long* p [[buffer(4)]],
    constant float* scalars [[buffer(5)]],
    const device MaskType* mask [[buffer(6)]],
    const device int* cu_seqlens_q [[buffer(7)]],  // Cumulative query sequence lengths
    const device int* cu_seqlens_k [[buffer(8)]],  // Cumulative key sequence lengths
    device float* lse [[buffer(9)]],
    const device int* table [[buffer(10)]], const device int* used [[buffer(11)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  AttnParams data = {};
  thread AttnParams* params = &data;
  params->H=p[HQ]; params->D=p[DQ]; params->gqa_factor=p[HQ]/p[HK];
  params->scale=scalars[0]; params->softcapping=scalars[1];
  const constant AttnMaskParams* mask_params = nullptr;
  // Pacifying compiler
  (void)lid;

  // Flash Attention variable-length indexing
  // tid.z is now the sequence index within the batch
  int batch_idx = tid.z;
  int head_idx = tid.y;
  int block_idx = tid.x;
  
  // Get sequence boundaries from cumulative lengths
  int q_seq_start = cu_seqlens_q[batch_idx*p[CQ0]];
  int q_seq_end = cu_seqlens_q[(batch_idx + 1)*p[CQ0]];
  int k_seq_start = p[CUK] ? cu_seqlens_k[batch_idx*p[CK0]] : 0;
  int k_seq_end = p[USED] ? k_seq_start + used[batch_idx*p[U0]] : cu_seqlens_k[(batch_idx + 1)*p[CK0]];
  
  int q_seq_len = q_seq_end - q_seq_start;
  int k_seq_len = k_seq_end - k_seq_start;
  
  // Check if this block is within the sequence
  if (block_idx * BQ >= q_seq_len) {
    return;
  }
  
  // Calculate offsets in the packed tensor format
  // Q/O shape: [total_tokens, num_heads, head_dim]
  // K/V shape: [total_tokens, num_heads_kv, head_dim]
  int q_offset = q_seq_start + block_idx * BQ;
  int k_offset = p[PAGED] ? 0 : k_seq_start;
  
  ulong kv_head_idx = head_idx / params->gqa_factor;
  
  // Move pointers to the correct position in packed format
  Q += q_offset * p[QS0] + head_idx * p[QS1];
  K += k_offset * p[KS1] + kv_head_idx * p[KS2];
  V += k_offset * p[VS1] + kv_head_idx * p[VS2];
  O += q_offset * p[OS0] + head_idx * p[OS1];
  
  if (has_mask) {
    // Mask indexing would need to be updated based on the mask format
    mask += batch_idx * mask_params->M_strides[0] + 
            head_idx * mask_params->M_strides[1];
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  // For packed tensors, stride between tokens is H * D
  int q_stride = p[QS0];
  int kv_stride = p[KS1];
  
  QBlockLoader loader_q(
      Q, q_stride, Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, kv_stride, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, p[VS1], Vs, simd_group_id, simd_lane_id);

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks apply scale
  int q_block_end = min(block_idx * BQ + BQ, q_seq_len);
  int q_block_size = q_block_end - block_idx * BQ;
  
  if (q_block_size < BQ) {
    loader_q.load_safe(short2(BD, q_block_size));
  } else {
    loader_q.load_unsafe();
  }
  // Scale the FP32 dot products, not the staged FP16/BF16 queries. Scaling
  // queries before staging would round once more and perturb sharp softmaxes.

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::min;
  }

  // Calculate number of K blocks for this sequence.
  // In general, we want to iterate over all key blocks.  However,
  // when causal masking is enabled we only need to process up to the
  // last key that influences this query block.  In decode mode
  // (q_seq_len < k_seq_len), the single query token logically sits
  // at the end of the key sequence.  Without adjusting for this the
  // causal computation would incorrectly restrict processing to only
  // the first key block, because the query position would appear to
  // be at index 0.  To handle this we compute a causal_offset that
  // shifts the query indices so they align with the end of the key
  // sequence when q_seq_len < k_seq_len.
  int kb_lim = (k_seq_len + BK - 1) / BK;

  if (do_causal) {
    // Offset the row indices for causal masking when the query length
    // is smaller than the key length (decode mode).  This ensures
    // that the computed row positions correspond to the correct
    // positions within the key sequence.
    int causal_offset = 0;
    if (q_seq_len != k_seq_len) {
      causal_offset = k_seq_len - q_seq_len;
    }

    // Determine the start/end of the current query block in the
    // (possibly offset) sequence.  The block index operates on
    // query positions but causal_offset places it relative to the
    // key positions when in decode mode.
    int q_block_start_in_seq = block_idx * BQ + causal_offset;
    int q_block_end_in_seq = q_block_start_in_seq + q_block_size;

    // Limit the number of key blocks so that blocks that are strictly
    // beyond the last valid key (for this row) are not processed.
    // When causal_offset > 0 this prevents prematurely exiting after
    // the first block in decode mode.
    kb_lim = min(kb_lim, (q_block_end_in_seq + BK - 1) / BK);
  }

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    if (p[PAGED]) {
      int logical = kb * BK;
      int physical = table[batch_idx*p[BT0] + (logical/p[PAGE])*p[BT1]];
      loader_k.src = K + physical*p[KS0] + (logical%p[PAGE])*p[KS1]
          + loader_k.bi*p[KS1] + loader_k.bj;
      loader_v.src = V + physical*p[VS0] + (logical%p[PAGE])*p[VS1]
          + loader_v.bi*p[VS1] + loader_v.bj;
    }
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int k_block_end = min(kb * BK + BK, k_seq_len);
    int k_block_size = k_block_end - kb * BK;
    
    if (k_block_size < BK) {
      loader_k.load_safe(short2(BD, k_block_size));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Softcap operates on natural scores before masking. Applying tanh after
    // the mask would turn -infinity into a finite, attended score.
    if (do_softcap) {
      using score_tile = decltype(Stile);
      STEEL_PRAGMA_UNROLL
      for (short i=0;i<score_tile::kTileRows;++i) {
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<score_tile::kTileCols;++j) {
          STEEL_PRAGMA_UNROLL
          for(short e=0;e<score_tile::MMAFrag_t::kElemsPerFrag;++e)
            Stile.frag_at(i,j)[e]=params->softcapping * precise::tanh(
                Stile.frag_at(i,j)[e]*params->scale/params->softcapping)*1.44269504089f;
        }
      }
    } else {
      AccumType score_scale[kRowsPT];
      STEEL_PRAGMA_UNROLL
      for (short i=0; i<kRowsPT; ++i) score_scale[i]=params->scale*1.44269504089f;
      Stile.template row_bin_op<MulOp>(score_scale);
    }

    // Mask out length sequence
    if (k_block_size < BK) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= k_block_size) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        // Compute row position for causal mask.  In decode mode
        // (q_seq_len < k_seq_len) the single query row should be
        // aligned with the end of the key sequence.  Without this
        // offset the row index would be zero and all but the first
        // key block would be erroneously masked out.
        int row_pos_causal = block_idx * BQ + tm + sm + (i * stile_t::kFragRows);
        if (q_seq_len != k_seq_len) {
          row_pos_causal += (k_seq_len - q_seq_len);
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos_in_seq = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos_causal < (col_pos_in_seq + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        // Use sequence-local positions
        const int row_pos_in_seq = block_idx * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos_in_seq = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int(mask_params->M_strides[2]),
              Int<1>{},
              q_seq_len,
              k_seq_len,
              row_pos_in_seq,  // Already sequence-local
              col_pos_in_seq); // Already sequence-local

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (k_block_size < BK) {
      loader_v.load_safe(short2(BD, k_block_size));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = isfinite(new_max[i]) ? fast::exp2(max_score[i] - new_max[i]) : 0.0f;
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  int output_row=q_seq_start+block_idx*BQ+tm+sm;
  if (sn == 0 && block_idx*BQ+tm+sm < q_seq_len)
    lse[head_idx*p[::T]+output_row]=sum_score[0]>0
        ? max_score[0]*0.6931471805599453f+log(sum_score[0]) : -INFINITY;
  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  // O is already pointing to the correct block position from earlier adjustment
  // Just need to offset within the block for this thread's tile
  device T* O_tile = O + (tm + sm) * p[OS0] + sn;

  if (q_block_size < BQ) {
    // Only store if this thread's tile is within the valid range
    if ((tm + sm) < q_block_size && sn < BD) {
      auto dst_tile_dims = short2(BD - sn, q_block_size - (tm + sm));
      Otile.template store_safe<T, 1, 1>(O_tile, p[OS0], dst_tile_dims);
    }
  } else {
    Otile.template store<T, 1, 1>(O_tile, p[OS0]);
  }
}

// clang-format off


template [[host_name("fast_attention")]] [[kernel]] decltype(attention<__TYPE__, __BQ__, __BK__, __D__, __WM__, 1, bool, float>) attention<__TYPE__, __BQ__, __BK__, __D__, __WM__, 1, bool, float>;
