import React, { useState } from 'react';
import { Activity, Clock, ShieldAlert, Cpu, Layers, HelpCircle, Flame, FileDown } from 'lucide-react';
import { ProfilingResult, KernelExecutionConfig, GPU_CATALOG } from '../core/gpu_simulation';
import { TechnicalReportModal } from './TechnicalReportModal';

interface NsightProfilerPanelProps {
  config: KernelExecutionConfig;
  eagerResult: ProfilingResult;
  tritonResult: ProfilingResult;
  cudaResult: ProfilingResult;
  inductorResult: ProfilingResult;
}

export const NsightProfilerPanel: React.FC<NsightProfilerPanelProps> = ({
  config,
  eagerResult,
  tritonResult,
  cudaResult,
  inductorResult,
}) => {
  const [activeProfilerTab, setActiveProfilerTab] = useState<'nsys' | 'ncu'>('nsys');
  const [selectedStallInfo, setSelectedStallInfo] = useState<string | null>(null);
  const [showReportModal, setShowReportModal] = useState<boolean>(false);

  const gpu = GPU_CATALOG[config.gpuId];

  // Stall categories
  const stallDescriptions: Record<string, { title: string; desc: string; fix: string }> = {
    longScoreboard: {
      title: 'Long Scoreboard Stall (Waiting for Global DRAM)',
      desc: 'Warp is stalled waiting for high-latency global memory data (LDG/STG to HBM/GDDR6X). Typical penalty is 400–800 clock cycles per cache miss.',
      fix: 'Fixed by operator fusion: keeping intermediate activations in fast SRAM and registers eliminates DRAM round-trips.',
    },
    shortScoreboard: {
      title: 'Short Scoreboard Stall (MIO / L1 / L2 Latency)',
      desc: 'Warp is stalled waiting for Shared Memory (LDS/STS) or texture/cache memory pipe response (~20-100 cycles).',
      fix: 'Addressed via vectorized 128-bit loads (float4) and conflict-free shared memory bank indexing.',
    },
    mathThrottle: {
      title: 'Math Pipe Throttle (ALU Pipeline Saturated)',
      desc: 'Warp scheduler cannot issue instructions because the floating-point / tensor math execution pipelines are saturated.',
      fix: 'This is actually desirable: in our fused kernel, math pipe utilization rises from 5.4% to 18.2%, proving the GPU is doing real compute rather than starving on memory.',
    },
    syncBarrier: {
      title: 'Barrier Synchronization Stall (__syncthreads / bar.sync)',
      desc: 'Warps waiting at a threadblock barrier for neighboring warps to reach the same program point.',
      fix: 'CUDA C++ implementation uses __shfl_down_sync intra-warp register shuffles, drastically cutting reliance on heavy __syncthreads().',
    },
  };

  return (
    <div className="space-y-6">
      {/* Header bar with Nsight profile switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
              NVIDIA NSIGHT SYSTEMS &amp; COMPUTE
            </span>
            <span className="text-xs text-slate-400">·</span>
            <span className="text-xs text-slate-500 font-mono">{gpu.name} ({gpu.computeCapability})</span>
          </div>
          <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 mt-0.5">
            Kernel Execution Path &amp; Warp Pipeline Stall Diagnostics
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setShowReportModal(true)}
            aria-label="Download Full Report"
            className="px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-750 hover:border-emerald-500 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap shadow-xs"
          >
            <FileDown className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            <span>Download Full Report</span>
          </button>

          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
            <button
              onClick={() => setActiveProfilerTab('nsys')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeProfilerTab === 'nsys'
                  ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              Nsight Systems (Timeline Trace)
            </button>
            <button
              onClick={() => setActiveProfilerTab('ncu')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeProfilerTab === 'ncu'
                  ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              Nsight Compute (Warp Stalls)
            </button>
          </div>
        </div>
      </div>

      {/* Full Technical Report Modal */}
      <TechnicalReportModal
        isOpen={showReportModal}
        onClose={() => setShowReportModal(false)}
        config={config}
        eagerResult={eagerResult}
        inductorResult={inductorResult}
        tritonResult={tritonResult}
        cudaResult={cudaResult}
      />

      {/* Tab 1: Nsight Systems Timeline Trace */}
      {activeProfilerTab === 'nsys' && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
            <div>
              <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <Clock className="w-4 h-4 text-emerald-600" />
                CUDA Stream Execution Timeline (nsys trace)
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Microsecond-accurate profiling trace showing kernel launch bubbles, CUDA API dispatches, and kernel span durations.
              </p>
            </div>
            <div className="text-xs font-mono text-slate-400">
              Timescale: Normalized μs
            </div>
          </div>

          {/* Trace Rows */}
          <div className="space-y-6 font-mono text-xs">
            {/* 1. PyTorch Eager Stream */}
            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-750">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                  <span className="font-bold text-slate-800 dark:text-slate-200">
                    PyTorch Eager: CUDA Stream 7 [Unfused]
                  </span>
                </div>
                <span className="text-rose-600 dark:text-rose-400 font-bold">
                  Total Latency: {eagerResult.latencyUs.toLocaleString()} μs (3 launches + 2 gaps)
                </span>
              </div>

              {/* Graphical timeline bar */}
              <div className="h-10 bg-slate-200 dark:bg-slate-800 rounded-lg p-1 flex gap-1 items-center overflow-x-auto">
                {/* Kernel 1: Add */}
                <div
                  style={{ flex: 25 }}
                  className="h-full bg-rose-500/80 hover:bg-rose-500 text-white rounded flex items-center justify-center text-[11px] px-2 whitespace-nowrap cursor-pointer transition-colors"
                  title="aten::add kernel execution (DRAM read x, res -> write add_out)"
                >
                  aten::add [{(eagerResult.latencyUs * 0.25).toFixed(1)} μs]
                </div>
                {/* Launch Bubble 1 */}
                <div
                  style={{ flex: 5 }}
                  className="h-full bg-amber-400/40 text-amber-800 dark:text-amber-300 rounded flex items-center justify-center text-[10px] whitespace-nowrap cursor-help"
                  title="Driver Launch Bubble (CPU host dispatch gap ~4.2 μs)"
                >
                  gap (~4μs)
                </div>
                {/* Kernel 2: LayerNorm */}
                <div
                  style={{ flex: 45 }}
                  className="h-full bg-rose-600/80 hover:bg-rose-600 text-white rounded flex items-center justify-center text-[11px] px-2 whitespace-nowrap cursor-pointer transition-colors"
                  title="aten::layer_norm kernel execution (DRAM read add_out -> 2-pass reduce -> write norm_out)"
                >
                  aten::layer_norm [{(eagerResult.latencyUs * 0.45).toFixed(1)} μs]
                </div>
                {/* Launch Bubble 2 */}
                <div
                  style={{ flex: 5 }}
                  className="h-full bg-amber-400/40 text-amber-800 dark:text-amber-300 rounded flex items-center justify-center text-[10px] whitespace-nowrap cursor-help"
                  title="Driver Launch Bubble (CPU host dispatch gap ~4.5 μs)"
                >
                  gap (~4μs)
                </div>
                {/* Kernel 3: SiLU */}
                <div
                  style={{ flex: 20 }}
                  className="h-full bg-rose-500/80 hover:bg-rose-500 text-white rounded flex items-center justify-center text-[11px] px-2 whitespace-nowrap cursor-pointer transition-colors"
                  title="aten::silu kernel execution (DRAM read norm_out -> write final)"
                >
                  aten::silu [{(eagerResult.latencyUs * 0.20).toFixed(1)} μs]
                </div>
              </div>
              <div className="flex justify-between text-[11px] text-slate-400 mt-1.5">
                <span>0 μs</span>
                <span className="text-rose-500 font-medium">3 dispatches · High memory wait time</span>
                <span>{eagerResult.latencyUs} μs</span>
              </div>
            </div>

            {/* 2. TorchScript / Inductor */}
            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-750">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  <span className="font-bold text-slate-800 dark:text-slate-200">
                    PyTorch 2.0 TorchCompile / Inductor [Partial Fusion]
                  </span>
                </div>
                <span className="text-blue-600 dark:text-blue-400 font-bold">
                  Total Latency: {inductorResult.latencyUs.toLocaleString()} μs ({inductorResult.speedupVsEager}x speedup)
                </span>
              </div>

              {/* Graphical timeline bar */}
              <div className="h-10 bg-slate-200 dark:bg-slate-800 rounded-lg p-1 flex gap-1 items-center overflow-x-auto">
                <div
                  style={{ flex: 60 }}
                  className="h-full bg-blue-500/80 hover:bg-blue-500 text-white rounded flex items-center justify-center text-[11px] px-2 whitespace-nowrap"
                >
                  inductor_fused_add_layernorm [{(inductorResult.latencyUs * 0.65).toFixed(1)} μs]
                </div>
                <div
                  style={{ flex: 5 }}
                  className="h-full bg-amber-400/40 text-amber-800 dark:text-amber-300 rounded flex items-center justify-center text-[10px] whitespace-nowrap"
                >
                  gap
                </div>
                <div
                  style={{ flex: 35 }}
                  className="h-full bg-blue-600/80 hover:bg-blue-600 text-white rounded flex items-center justify-center text-[11px] px-2 whitespace-nowrap"
                >
                  inductor_silu [{(inductorResult.latencyUs * 0.35).toFixed(1)} μs]
                </div>
              </div>
            </div>

            {/* 3. OpenAI Triton Fused Kernel */}
            <div className="p-3.5 rounded-xl bg-emerald-50/40 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  <span className="font-bold text-slate-800 dark:text-slate-200">
                    OpenAI Triton: _fused_add_layernorm_silu_kernel [1 Launch]
                  </span>
                </div>
                <span className="text-emerald-600 dark:text-emerald-400 font-bold">
                  Total Latency: {tritonResult.latencyUs.toLocaleString()} μs ({tritonResult.speedupVsEager}x speedup)
                </span>
              </div>

              {/* Graphical timeline bar */}
              <div className="h-10 bg-slate-200 dark:bg-slate-800 rounded-lg p-1 flex gap-1 items-center">
                <div
                  style={{ flex: (tritonResult.latencyUs / eagerResult.latencyUs) * 100 }}
                  className="h-full bg-emerald-600 hover:bg-emerald-500 text-white rounded flex items-center justify-between px-3 text-[11px] whitespace-nowrap transition-colors"
                >
                  <span>_fused_add_layernorm_silu_kernel (Triton)</span>
                  <span className="font-bold">{tritonResult.latencyUs} μs</span>
                </div>
                <div
                  style={{ flex: 100 - (tritonResult.latencyUs / eagerResult.latencyUs) * 100 }}
                  className="h-full border border-dashed border-emerald-400/40 rounded flex items-center justify-center text-[11px] text-emerald-600 dark:text-emerald-400 bg-emerald-50/50 dark:bg-emerald-950/30"
                >
                  Saved {(eagerResult.latencyUs - tritonResult.latencyUs).toLocaleString()} μs (Latency Eliminated)
                </div>
              </div>
            </div>

            {/* 4. Custom CUDA C++ Kernel */}
            <div className="p-3.5 rounded-xl bg-emerald-50/40 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  <span className="font-bold text-slate-800 dark:text-slate-200">
                    CUDA C++: FusedAddLayerNormSiLUKernel [1 Launch]
                  </span>
                </div>
                <span className="text-emerald-600 dark:text-emerald-400 font-bold">
                  Total Latency: {cudaResult.latencyUs.toLocaleString()} μs ({cudaResult.speedupVsEager}x speedup)
                </span>
              </div>

              {/* Graphical timeline bar */}
              <div className="h-10 bg-slate-200 dark:bg-slate-800 rounded-lg p-1 flex gap-1 items-center">
                <div
                  style={{ flex: (cudaResult.latencyUs / eagerResult.latencyUs) * 100 }}
                  className="h-full bg-teal-600 hover:bg-teal-500 text-white rounded flex items-center justify-between px-3 text-[11px] whitespace-nowrap transition-colors"
                >
                  <span>FusedAddLayerNormSiLUKernel (__shfl_down_sync)</span>
                  <span className="font-bold">{cudaResult.latencyUs} μs</span>
                </div>
                <div
                  style={{ flex: 100 - (cudaResult.latencyUs / eagerResult.latencyUs) * 100 }}
                  className="h-full border border-dashed border-teal-400/40 rounded flex items-center justify-center text-[11px] text-teal-600 dark:text-teal-400 bg-teal-50/50 dark:bg-teal-950/30"
                >
                  Saved {(eagerResult.latencyUs - cudaResult.latencyUs).toLocaleString()} μs (Peak Bandwidth Saturation)
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Nsight Compute Warp State & Memory-Stall Pipeline */}
      {activeProfilerTab === 'ncu' && (
        <div className="space-y-6">
          {/* Warp Occupancy Meters */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'PyTorch Eager', occupancy: eagerResult.warpOccupancyPercent, color: 'text-rose-500', barColor: 'bg-rose-500' },
              { label: 'Inductor Fused', occupancy: inductorResult.warpOccupancyPercent, color: 'text-blue-500', barColor: 'bg-blue-500' },
              { label: 'Custom Triton Kernel', occupancy: tritonResult.warpOccupancyPercent, color: 'text-emerald-500', barColor: 'bg-emerald-500' },
              { label: 'Custom CUDA C++', occupancy: cudaResult.warpOccupancyPercent, color: 'text-teal-500', barColor: 'bg-teal-500' },
            ].map((item) => (
              <div
                key={item.label}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs"
              >
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{item.label}</div>
                <div className="flex items-baseline justify-between mb-2">
                  <span className={`text-2xl font-bold font-mono ${item.color}`}>
                    {item.occupancy}%
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    Max: {gpu.maxWarpsPerSm} warps/SM
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${item.barColor} transition-all duration-500`}
                    style={{ width: `${item.occupancy}%` }}
                  />
                </div>
                <div className="text-[11px] text-slate-400 mt-2 font-mono">
                  {Math.round((item.occupancy / 100) * gpu.maxWarpsPerSm)} active warps per SM
                </div>
              </div>
            ))}
          </div>

          {/* Warp Issue Stall Pipeline Breakdown Chart */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800 gap-2">
              <div>
                <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-600" />
                  Warp Issue Stall Breakdown (smsp__warp_issue_stalled_*)
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Percentage of cycles in which eligible warps could not be selected by the warp scheduler.
                </p>
              </div>
              <div className="text-xs text-slate-400">
                Click any stall category for hardware diagnostics
              </div>
            </div>

            {/* Stacked Comparative Bars */}
            <div className="space-y-5 font-mono text-xs">
              {[
                { title: 'PyTorch Eager', data: eagerResult.stallBreakdown },
                { title: 'TorchCompile / Inductor', data: inductorResult.stallBreakdown },
                { title: 'Custom Triton Kernel', data: tritonResult.stallBreakdown },
                { title: 'Custom CUDA C++', data: cudaResult.stallBreakdown },
              ].map((engine) => (
                <div key={engine.title}>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="font-semibold text-slate-800 dark:text-slate-200">{engine.title}</span>
                    <span className="text-[11px] text-slate-500">
                      Long Scoreboard: <strong>{engine.data.longScoreboardMemStall}%</strong> · Active: <strong>{engine.data.activeExecuting}%</strong>
                    </span>
                  </div>
                  {/* Multi-segment bar */}
                  <div className="h-8 rounded-lg overflow-hidden flex bg-slate-200 dark:bg-slate-800">
                    {/* Long Scoreboard (DRAM) */}
                    <div
                      style={{ width: `${engine.data.longScoreboardMemStall}%` }}
                      onClick={() => setSelectedStallInfo('longScoreboard')}
                      className="h-full bg-rose-500 hover:opacity-90 cursor-pointer flex items-center justify-center text-white text-[10px] truncate px-1 transition-opacity"
                      title={`Long Scoreboard (DRAM Stall): ${engine.data.longScoreboardMemStall}%`}
                    >
                      {engine.data.longScoreboardMemStall > 15 ? `DRAM ${engine.data.longScoreboardMemStall}%` : ''}
                    </div>
                    {/* Short Scoreboard (MIO/L1) */}
                    <div
                      style={{ width: `${engine.data.shortScoreboardMIO}%` }}
                      onClick={() => setSelectedStallInfo('shortScoreboard')}
                      className="h-full bg-amber-500 hover:opacity-90 cursor-pointer flex items-center justify-center text-white text-[10px] truncate px-1 transition-opacity"
                      title={`Short Scoreboard (MIO/L1): ${engine.data.shortScoreboardMIO}%`}
                    >
                      {engine.data.shortScoreboardMIO > 10 ? `MIO ${engine.data.shortScoreboardMIO}%` : ''}
                    </div>
                    {/* Math Pipe Throttle */}
                    <div
                      style={{ width: `${engine.data.mathPipeThrottle}%` }}
                      onClick={() => setSelectedStallInfo('mathThrottle')}
                      className="h-full bg-indigo-500 hover:opacity-90 cursor-pointer flex items-center justify-center text-white text-[10px] truncate px-1 transition-opacity"
                      title={`Math Pipe Throttle: ${engine.data.mathPipeThrottle}%`}
                    >
                      {engine.data.mathPipeThrottle > 10 ? `Math ${engine.data.mathPipeThrottle}%` : ''}
                    </div>
                    {/* Sync Barrier */}
                    <div
                      style={{ width: `${engine.data.syncBarrier}%` }}
                      onClick={() => setSelectedStallInfo('syncBarrier')}
                      className="h-full bg-slate-400 dark:bg-slate-600 hover:opacity-90 cursor-pointer flex items-center justify-center text-white text-[10px] truncate px-1 transition-opacity"
                      title={`Barrier Sync: ${engine.data.syncBarrier}%`}
                    >
                      {engine.data.syncBarrier > 8 ? `Sync` : ''}
                    </div>
                    {/* Active Instruction Issue */}
                    <div
                      style={{ width: `${engine.data.activeExecuting}%` }}
                      className="h-full bg-emerald-500 hover:opacity-90 cursor-default flex items-center justify-center text-white text-[10px] truncate px-1 font-bold"
                      title={`Active Executing Instruction: ${engine.data.activeExecuting}%`}
                    >
                      Active {engine.data.activeExecuting}%
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-slate-100 dark:border-slate-800 text-xs">
              <button
                onClick={() => setSelectedStallInfo('longScoreboard')}
                className="flex items-center gap-1.5 hover:underline"
              >
                <span className="w-3 h-3 rounded bg-rose-500 shrink-0" />
                <span className="text-slate-600 dark:text-slate-300">Long Scoreboard (DRAM Stall)</span>
              </button>
              <button
                onClick={() => setSelectedStallInfo('shortScoreboard')}
                className="flex items-center gap-1.5 hover:underline"
              >
                <span className="w-3 h-3 rounded bg-amber-500 shrink-0" />
                <span className="text-slate-600 dark:text-slate-300">Short Scoreboard (MIO/L1)</span>
              </button>
              <button
                onClick={() => setSelectedStallInfo('mathThrottle')}
                className="flex items-center gap-1.5 hover:underline"
              >
                <span className="w-3 h-3 rounded bg-indigo-500 shrink-0" />
                <span className="text-slate-600 dark:text-slate-300">Math Pipe Throttle</span>
              </button>
              <button
                onClick={() => setSelectedStallInfo('syncBarrier')}
                className="flex items-center gap-1.5 hover:underline"
              >
                <span className="w-3 h-3 rounded bg-slate-400 dark:bg-slate-600 shrink-0" />
                <span className="text-slate-600 dark:text-slate-300">Barrier Synchronization</span>
              </button>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-emerald-500 shrink-0" />
                <span className="text-emerald-700 dark:text-emerald-400 font-semibold">Active Instruction Issue</span>
              </div>
            </div>

            {/* Stall Deep Dive Modal / Drawer */}
            {selectedStallInfo && (
              <div className="p-4 rounded-xl bg-slate-900 text-slate-100 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="font-bold text-sm text-white flex items-center gap-2">
                    <HelpCircle className="w-4 h-4 text-emerald-400" />
                    {stallDescriptions[selectedStallInfo].title}
                  </h4>
                  <button
                    onClick={() => setSelectedStallInfo(null)}
                    className="text-xs text-slate-400 hover:text-white"
                  >
                    Close
                  </button>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {stallDescriptions[selectedStallInfo].desc}
                </p>
                <div className="p-2.5 rounded bg-emerald-950/60 border border-emerald-800 text-xs text-emerald-300">
                  <strong>Engineering Fix in Kernel:</strong> {stallDescriptions[selectedStallInfo].fix}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
