import React, { useState } from 'react';
import { ArrowDown, ArrowUp, HardDrive, Cpu, Zap, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { ProfilingResult, KernelExecutionConfig, getBytesPerElement } from '../core/gpu_simulation';

interface MemoryPipelineVisualizerProps {
  config: KernelExecutionConfig;
  eagerResult: ProfilingResult;
  tritonResult: ProfilingResult;
  cudaResult: ProfilingResult;
}

export const MemoryPipelineVisualizer: React.FC<MemoryPipelineVisualizerProps> = ({
  config,
  eagerResult,
  tritonResult,
  cudaResult,
}) => {
  const [selectedView, setSelectedView] = useState<'sideBySide' | 'eager' | 'fused'>('sideBySide');

  const bytesPerElem = getBytesPerElement(config.precision);
  const tensorBytes = config.batchSize * config.seqLength * config.hiddenDim * bytesPerElem;
  const tensorMB = (tensorBytes / (1024 * 1024)).toFixed(2);

  return (
    <div className="space-y-6">
      {/* Top Banner explaining the architectural problem and solution */}
      <div className="bg-slate-900 text-slate-100 rounded-xl p-5 border border-slate-800">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                GPU Memory Hierarchy Optimization
              </span>
              <span className="text-xs text-slate-400">Transformer Feed-Forward / Attention Block</span>
            </div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              Eliminating Memory Bandwidth Bottlenecks via Operator Fusion
            </h2>
            <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Standard PyTorch Eager computes <code>x = silu(layernorm(x + res))</code> as 3 isolated ATen kernels.
              Because global HBM/GDDR6X memory is ~100x higher latency than on-chip registers and shared memory (SRAM),
              the GPU spends over 64% of clock cycles stalled on the memory bus (Long Scoreboard stalls).
              Our fused Triton and CUDA C++ kernels fuse all 3 operations into a single kernel launch.
            </p>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-1 bg-slate-800 p-1 rounded-lg shrink-0">
            <button
              onClick={() => setSelectedView('sideBySide')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                selectedView === 'sideBySide'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Side-by-Side Comparison
            </button>
            <button
              onClick={() => setSelectedView('eager')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                selectedView === 'eager'
                  ? 'bg-rose-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Unfused Eager Path
            </button>
            <button
              onClick={() => setSelectedView('fused')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                selectedView === 'fused'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Fused Kernel Path
            </button>
          </div>
        </div>
      </div>

      {/* Visual Pipeline Comparison Grids */}
      <div className={`grid gap-6 ${selectedView === 'sideBySide' ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
        {/* 1. Unfused PyTorch Eager Pipeline */}
        {(selectedView === 'sideBySide' || selectedView === 'eager') && (
          <div className="bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-950/60 rounded-xl p-5 shadow-xs">
            <div className="flex items-center justify-between pb-3 border-b border-rose-100 dark:border-rose-900/30">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100">
                  Unfused PyTorch Eager Pipeline (3 DRAM Round-Trips)
                </h3>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-rose-50 dark:bg-rose-950/50 text-rose-700 dark:text-rose-400">
                {eagerResult.dramTrafficMB} MB DRAM Traffic
              </span>
            </div>

            <div className="mt-5 space-y-4">
              {/* Step 1: Residual Add */}
              <div className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
                <div className="flex items-center justify-between text-xs font-mono mb-2">
                  <span className="font-semibold text-slate-800 dark:text-slate-200">1. Kernel aten::add(x, res)</span>
                  <span className="text-rose-600 dark:text-rose-400">DRAM Round-Trip 1</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-[11px] text-center font-mono">
                  <div className="p-1.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300">
                    Read X ({tensorMB} MB)
                  </div>
                  <div className="p-1.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300">
                    Read RES ({tensorMB} MB)
                  </div>
                  <div className="p-1.5 rounded bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-400">
                    Write add_out ({tensorMB} MB)
                  </div>
                </div>
              </div>

              {/* DRAM Intermediary Callout 1 */}
              <div className="flex items-center justify-center gap-2 text-xs font-mono text-rose-600 dark:text-rose-400 py-0.5">
                <ArrowDown className="w-3.5 h-3.5 animate-bounce" />
                <span>Intermediate Tensor flushed to Global Memory HBM (High Latency)</span>
                <ArrowDown className="w-3.5 h-3.5 animate-bounce" />
              </div>

              {/* Step 2: LayerNorm */}
              <div className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
                <div className="flex items-center justify-between text-xs font-mono mb-2">
                  <span className="font-semibold text-slate-800 dark:text-slate-200">2. Kernel aten::layer_norm(add_out)</span>
                  <span className="text-rose-600 dark:text-rose-400">DRAM Round-Trip 2</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] text-center font-mono">
                  <div className="p-1.5 rounded bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-400">
                    Read add_out from DRAM ({tensorMB} MB)
                  </div>
                  <div className="p-1.5 rounded bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-400">
                    Write norm_out to DRAM ({tensorMB} MB)
                  </div>
                </div>
              </div>

              {/* DRAM Intermediary Callout 2 */}
              <div className="flex items-center justify-center gap-2 text-xs font-mono text-rose-600 dark:text-rose-400 py-0.5">
                <ArrowDown className="w-3.5 h-3.5 animate-bounce" />
                <span>Second flush to HBM + CPU Host Driver Launch Gap (~4.2 μs)</span>
                <ArrowDown className="w-3.5 h-3.5 animate-bounce" />
              </div>

              {/* Step 3: SiLU */}
              <div className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
                <div className="flex items-center justify-between text-xs font-mono mb-2">
                  <span className="font-semibold text-slate-800 dark:text-slate-200">3. Kernel aten::silu(norm_out)</span>
                  <span className="text-slate-500">Final Store</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] text-center font-mono">
                  <div className="p-1.5 rounded bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-400">
                    Read norm_out ({tensorMB} MB)
                  </div>
                  <div className="p-1.5 rounded bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400">
                    Write Final Output ({tensorMB} MB)
                  </div>
                </div>
              </div>

              {/* Unfused Summary box */}
              <div className="p-3 rounded-lg bg-rose-50/70 dark:bg-rose-950/30 text-rose-900 dark:text-rose-300 text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
                <div>
                  <div className="font-semibold">Inefficiency Bottleneck:</div>
                  Total DRAM Traffic = <strong>~7.2x</strong> input tensor size. 3 separate CUDA kernel launches create host sync gaps and saturate GPU memory channels with disposable intermediate tensors.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 2. Custom Fused Kernel Pipeline */}
        {(selectedView === 'sideBySide' || selectedView === 'fused') && (
          <div className="bg-white dark:bg-slate-900 border border-emerald-200 dark:border-emerald-950/60 rounded-xl p-5 shadow-xs">
            <div className="flex items-center justify-between pb-3 border-b border-emerald-100 dark:border-emerald-900/30">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100">
                  Custom Fused Kernel Pipeline (Zero Intermediate DRAM Writes)
                </h3>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-400">
                {tritonResult.dramTrafficMB} MB DRAM Traffic (-57%)
              </span>
            </div>

            <div className="mt-5 space-y-4">
              {/* Single Kernel Launch Box */}
              <div className="p-4 rounded-xl border border-emerald-200 dark:border-emerald-800/60 bg-emerald-50/30 dark:bg-emerald-950/20 space-y-3">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="font-bold text-emerald-700 dark:text-emerald-400">
                    _fused_add_layernorm_silu_kernel &lt;Single Dispatch&gt;
                  </span>
                  <span className="text-slate-500">1x Launch (~3.2 μs)</span>
                </div>

                {/* Internal Pipeline Stages */}
                <div className="space-y-2">
                  {/* Stage 1: Vectorized Coalesced Read */}
                  <div className="p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                    <div className="flex items-center justify-between text-[11px] font-mono text-slate-700 dark:text-slate-300 mb-1">
                      <span>Phase A: Vectorized 128-bit LDG.E.128</span>
                      <span className="text-emerald-600 dark:text-emerald-400">1 Single Read Phase</span>
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Coalesced streaming of <code>X</code> ({tensorMB} MB) and <code>RES</code> ({tensorMB} MB) directly into SM Registers.
                    </div>
                  </div>

                  {/* Stage 2: Register Arithmetic & SRAM Reduction */}
                  <div className="p-2.5 rounded-lg bg-emerald-100/60 dark:bg-emerald-900/30 border border-emerald-300 dark:border-emerald-700">
                    <div className="flex items-center justify-between text-[11px] font-mono text-emerald-900 dark:text-emerald-300 mb-1">
                      <span>Phase B: Warp Shuffle Reduction (__shfl_down_sync)</span>
                      <span className="font-semibold text-emerald-700 dark:text-emerald-400">On-Chip Registers / SRAM</span>
                    </div>
                    <div className="text-[11px] text-emerald-800 dark:text-emerald-300/90 space-y-1">
                      <div>• <code>y = x + res</code> computed and retained in hardware register file.</div>
                      <div>• Mean &amp; Variance two-pass reduction computed via 32-thread register shuffle without touching DRAM.</div>
                    </div>
                  </div>

                  {/* Stage 3: In-Register Affine + SiLU */}
                  <div className="p-2.5 rounded-lg bg-emerald-100/60 dark:bg-emerald-900/30 border border-emerald-300 dark:border-emerald-700">
                    <div className="flex items-center justify-between text-[11px] font-mono text-emerald-900 dark:text-emerald-300 mb-1">
                      <span>Phase C: In-Register Affine Normalization &amp; SiLU</span>
                      <span className="font-semibold text-emerald-700 dark:text-emerald-400">Zero Memory Latency</span>
                    </div>
                    <div className="text-[11px] text-emerald-800 dark:text-emerald-300/90">
                      <code>out = (diff * rstd * gamma + beta) * sigmoid(norm)</code> evaluated in ALU registers.
                    </div>
                  </div>

                  {/* Stage 4: Single Write */}
                  <div className="p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                    <div className="flex items-center justify-between text-[11px] font-mono text-slate-700 dark:text-slate-300 mb-1">
                      <span>Phase D: Single Coalesced STG.E.128 Write</span>
                      <span className="text-emerald-600 dark:text-emerald-400">1 Single Write Phase</span>
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Writes final output tensor ({tensorMB} MB) back to Global HBM.
                    </div>
                  </div>
                </div>
              </div>

              {/* Fused Summary box */}
              <div className="p-3 rounded-lg bg-emerald-50/70 dark:bg-emerald-950/30 text-emerald-900 dark:text-emerald-300 text-xs flex items-start gap-2">
                <CheckCircle className="w-4 h-4 shrink-0 mt-0.5 text-emerald-600" />
                <div>
                  <div className="font-semibold">Performance Outcome:</div>
                  DRAM Traffic drops from <strong>{eagerResult.dramTrafficMB} MB</strong> to <strong>{tritonResult.dramTrafficMB} MB</strong> (<strong>{tritonResult.bandwidthReductionPercent}%</strong> reduction).
                  Recovers <strong>38% VRAM throughput</strong>, accelerating execution by <strong>{tritonResult.speedupVsEager}x</strong>.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Hardware Memory Specs Reference Table */}
      <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/60 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <HardDrive className="w-4 h-4 text-slate-500" />
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-600 dark:text-slate-400">
            Hardware Latency &amp; Bandwidth Hierarchy (NVIDIA Ada Lovelace SM 8.9 / Ampere SM 8.0)
          </h4>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
            <div className="text-slate-400 text-[10px] uppercase">Registers (RF)</div>
            <div className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">~1 Cycle</div>
            <div className="text-[11px] text-emerald-600 dark:text-emerald-400 mt-1">&gt;20,000 GB/s bandwidth</div>
          </div>
          <div className="p-2.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
            <div className="text-slate-400 text-[10px] uppercase">Shared Memory (SRAM)</div>
            <div className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">~19 Cycles</div>
            <div className="text-[11px] text-emerald-600 dark:text-emerald-400 mt-1">~12,000 GB/s aggregate</div>
          </div>
          <div className="p-2.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
            <div className="text-slate-400 text-[10px] uppercase">L2 Cache</div>
            <div className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">~200 Cycles</div>
            <div className="text-[11px] text-slate-500 mt-1">~3,500 GB/s bandwidth</div>
          </div>
          <div className="p-2.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
            <div className="text-slate-400 text-[10px] uppercase">Global Memory (HBM / GDDR6X)</div>
            <div className="text-sm font-bold text-rose-600 dark:text-rose-400 mt-0.5">~400-800 Cycles</div>
            <div className="text-[11px] text-rose-600 dark:text-rose-400 mt-1">1,008 GB/s (RTX 4090)</div>
          </div>
        </div>
      </div>
    </div>
  );
};
