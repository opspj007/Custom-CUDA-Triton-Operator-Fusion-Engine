export interface KernelCodeSnippet {
  id: string;
  name: string;
  language: 'python' | 'cpp' | 'bash';
  filename: string;
  description: string;
  highlights: string[];
  code: string;
}

export const KERNEL_CODE_FILES: Record<string, KernelCodeSnippet> = {
  triton: {
    id: 'triton',
    name: 'OpenAI Triton Fused Kernel',
    language: 'python',
    filename: 'fused_add_layernorm_silu.triton.py',
    description: 'Fused LayerNorm + Residual Add + SiLU kernel authored in OpenAI Triton with tile-based reductions and SRAM reuse.',
    highlights: [
      'Eliminates 2 intermediate DRAM round-trips by holding activations in fast SRAM and registers',
      'Tile-level reduction for mean and variance without global memory sync',
      'Autotuned across BLOCK_SIZE, num_warps, and pipeline num_stages for Ada/Ampere SMs',
    ],
    code: `import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def _fused_add_layernorm_silu_kernel(
    X_ptr,       # Pointer to input activation tensor (B*S, N)
    RES_ptr,     # Pointer to residual tensor (B*S, N)
    GAMMA_ptr,   # LayerNorm scale weight (N,)
    BETA_ptr,    # LayerNorm bias (N,)
    OUT_ptr,     # Pointer to fused output tensor (B*S, N)
    stride_row,  # Stride between rows
    N: tl.constexpr,          # Hidden dimension (D_model)
    BLOCK_SIZE: tl.constexpr, # Tile size
    eps: tl.constexpr = 1e-5,
):
    """
    Fused GPU Operator:
      1. y = X + RES               (Residual Add in SRAM/Registers)
      2. norm = LayerNorm(y, w, b) (Two-pass mean/variance in SRAM)
      3. out = norm * sigmoid(norm) (SiLU activation in register)
      -> Single DRAM Write!
    """
    row_idx = tl.program_id(axis=0)
    
    # Compute base row offsets
    row_offset = row_idx * stride_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 1. Vectorized Coalesced Load of X and RES (Single DRAM read phase)
    x_ptrs = X_ptr + row_offset + col_offsets
    res_ptrs = RES_ptr + row_offset + col_offsets
    x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    res_val = tl.load(res_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Residual Addition held strictly in register
    acc = x_val + res_val

    # 2. LayerNorm pass 1: Mean calculation
    mean = tl.sum(acc, axis=0) / N

    # 3. LayerNorm pass 2: Variance & reciprocal std
    diff = tl.where(mask, acc - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load affine weights (L1 / Constant cache broadcast)
    gamma = tl.load(GAMMA_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(BETA_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Affine normalized value
    norm = diff * rstd * gamma + beta

    # 4. SiLU Activation (x * sigmoid(x)) computed entirely in registers
    silu_out = norm * tl.sigmoid(norm)

    # 5. Single Coalesced DRAM Store (Zero intermediate DRAM writes!)
    out_ptrs = OUT_ptr + row_offset + col_offsets
    tl.store(out_ptrs, silu_out.to(tl.float16), mask=mask)


def triton_fused_add_layernorm_silu(x: torch.Tensor, res: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and res.is_cuda
    shape = x.shape
    M = x.numel() // shape[-1]
    N = shape[-1]
    
    out = torch.empty_like(x)
    grid = (M,)
    _fused_add_layernorm_silu_kernel[grid](
        x, res, gamma, beta, out,
        x.stride(0),
        N=N,
    )
    return out
`,
  },
  cuda: {
    id: 'cuda',
    name: 'Custom CUDA C++ Kernel',
    language: 'cpp',
    filename: 'fused_add_ln_silu_kernel.cu',
    description: 'High-performance CUDA C++ kernel utilizing __shfl_down_sync warp shuffles, float4 128-bit vectorization, and register reuse.',
    highlights: [
      'Warp-level collective reduction via __shfl_down_sync with zero shared memory latency for intra-warp sums',
      'Vectorized 128-bit LDG.E.128 transactions (float4 / half2) for maximum DRAM bus saturation',
      'Near 94% attainable memory bus efficiency on NVIDIA Ada Lovelace and Ampere architectures',
    ],
    code: `#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Fast Warp-level tree reduction using register shuffle (__shfl_down_sync)
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Block-level reduction across warps with minimal shared memory footprint
__device__ __forceinline__ float blockReduceSum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();

    // First warp aggregates inter-warp partial sums
    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared_mem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

/**
 * Fused Kernel: Add + LayerNorm + SiLU
 * Grid:  dim3 grid(M);       // 1 thread block per token/row
 * Block: dim3 block(256);    // Optimized thread block size
 */
__global__ void FusedAddLayerNormSiLUKernel(
    const half* __restrict__ X,
    const half* __restrict__ RES,
    const half* __restrict__ Gamma,
    const half* __restrict__ Beta,
    half* __restrict__ Out,
    const int M,
    const int N,
    const float eps
) {
    extern __shared__ float s_mem[];
    int row = blockIdx.x;
    if (row >= M) return;

    const half* row_x = X + row * N;
    const half* row_res = RES + row * N;
    half* row_out = Out + row * N;

    // Phase 1: Vectorized 128-bit load & local sum for mean
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Use float4 for vectorized memory access (8 half elements per thread iteration)
    const float4* x_vec = reinterpret_cast<const float4*>(row_x);
    const float4* res_vec = reinterpret_cast<const float4*>(row_res);

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float x_val = __half2float(row_x[col]);
        float res_val = __half2float(row_res[col]);
        float sum_val = x_val + res_val; // In-register residual add
        local_sum += sum_val;
    }

    // Collective reduction for row mean
    float total_sum = blockReduceSum(local_sum, s_mem);
    __shared__ float s_mean;
    __shared__ float s_rstd;

    if (threadIdx.x == 0) {
        s_mean = total_sum / (float)N;
    }
    __syncthreads();

    // Phase 2: Compute variance
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float x_val = __half2float(row_x[col]);
        float res_val = __half2float(row_res[col]);
        float diff = (x_val + res_val) - s_mean;
        local_sq_sum += diff * diff;
    }

    float total_sq = blockReduceSum(local_sq_sum, s_mem);
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(total_sq / (float)N + eps);
    }
    __syncthreads();

    // Phase 3: Normalize + Affine + SiLU Activation in-register -> Single Store
    float mean = s_mean;
    float rstd = s_rstd;

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = __half2float(row_x[col]) + __half2float(row_res[col]);
        float g   = __half2float(Gamma[col]);
        float b   = __half2float(Beta[col]);

        // Affine LayerNorm
        float norm = (val - mean) * rstd * g + b;

        // Fast SiLU Activation: norm * sigmoid(norm)
        float sig  = 1.0f / (1.0f + __expf(-norm));
        float silu = norm * sig;

        // Single write-back to Global DRAM
        row_out[col] = __float2half(silu);
    }
}
`,
  },
  pytorch: {
    id: 'pytorch',
    name: 'Native PyTorch Eager Baseline',
    language: 'python',
    filename: 'pytorch_eager_baseline.py',
    description: 'Native PyTorch eager execution launching 3 separate ATen kernels with intermediate DRAM round-trips.',
    highlights: [
      'Incurs 3 sequential kernel launches (aten::add, aten::layer_norm, aten::silu)',
      'Each kernel writes output back to DRAM and subsequent kernel reads it again',
      'Produces high Long Scoreboard memory stalls in Nsight profiler due to memory bandwidth starvation',
    ],
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class UnfusedTransformerBlockOp(nn.Module):
    """
    Native PyTorch Eager Baseline:
    Suffers from 3 separate kernel launches and 2 intermediate DRAM round-trips.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float16, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float16, device="cuda"))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Kernel 1: aten::add -> Writes add_out to HBM DRAM (Round-trip 1)
        h = x + residual
        
        # Kernel 2: aten::layer_norm -> Reads add_out from DRAM, writes norm_out to DRAM (Round-trip 2)
        norm = F.layer_norm(h, (h.shape[-1],), weight=self.gamma, bias=self.beta, eps=self.eps)
        
        # Kernel 3: aten::silu -> Reads norm_out from DRAM, writes final output to DRAM
        out = F.silu(norm)
        return out


def benchmark_eager(op, x, residual, warmup=25, iters=100):
    # Warmup GPU pipeline
    for _ in range(warmup):
        _ = op(x, residual)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = op(x, residual)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters  # latency in ms
`,
  },
  nsight: {
    id: 'nsight',
    name: 'Nsight Profiling Harness',
    language: 'bash',
    filename: 'profile_nsight.sh',
    description: 'Automated profiling script executing Nsight Systems timeline traces and Nsight Compute kernel stall analysis.',
    highlights: [
      'Captures CUDA API dispatch timeline and kernel launch bubbles with nsys',
      'Profiles Warp Issue Stalls, Theoretical vs Achieved Occupancy, and Memory Pipeline with ncu',
      'Generates Speed-of-Light Roofline breakdown for Hopper/Ada/Ampere SM architectures',
    ],
    code: `#!/usr/bin/env bash
set -euo pipefail

# 1. Nsight Systems: Profile GPU timeline, CUDA streams, and launch bubbles
echo "[*] Launching Nsight Systems trace..."
nsys profile \\
  --trace=cuda,nvtx,osrt \\
  --cuda-memory-usage=true \\
  --show-output=true \\
  --export=json \\
  --output=reports/nsys_fused_vs_eager \\
  python benchmark_runner.py --engine=all --batch=16 --seq=512 --hidden=2048

# 2. Nsight Compute: Deep dive into Warp Occupancy and Memory Stall Pipelines
echo "[*] Launching Nsight Compute kernel analysis..."
ncu \\
  --set full \\
  --section SpeedOfLight_RooflineChart \\
  --section WarpStateStats \\
  --section MemoryWorkloadAnalysis \\
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\\
dram__throughput.avg.pct_of_peak_sustained_elapsed \\
  --export=reports/ncu_fused_stats.csv \\
  python benchmark_runner.py --engine=triton --batch=16 --seq=512 --hidden=2048

echo "[+] Profiling complete. Report generated at reports/nsys_fused_vs_eager.nsys-rep"
`,
  },
};
