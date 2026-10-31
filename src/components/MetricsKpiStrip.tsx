import React from 'react';
import { Zap, Database, Activity, RefreshCw, CheckCircle2 } from 'lucide-react';
import { ProfilingResult } from '../core/gpu_simulation';

interface MetricsKpiStripProps {
  eagerResult: ProfilingResult;
  tritonResult: ProfilingResult;
  cudaResult: ProfilingResult;
  inductorResult: ProfilingResult;
  selectedEngine: 'triton' | 'cuda';
}

export const MetricsKpiStrip: React.FC<MetricsKpiStripProps> = ({
  eagerResult,
  tritonResult,
  cudaResult,
  selectedEngine,
}) => {
  const activeFused = selectedEngine === 'triton' ? tritonResult : cudaResult;

  // Exact metrics referenced in project accomplishments
  const speedup = activeFused.speedupVsEager;
  const vramThroughputRecovery = activeFused.bandwidthReductionPercent;
  const dramTrafficSavedMB = (eagerResult.dramTrafficMB - activeFused.dramTrafficMB).toFixed(1);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
      {/* 1. Latency Speedup */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
          <span>Latency Speedup</span>
          <Zap className="w-4 h-4 text-amber-500" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
            {speedup}x
          </span>
          <span className="text-xs text-slate-400 font-mono">vs PyTorch Eager</span>
        </div>
        <div className="text-[11px] text-slate-500 mt-1 font-mono">
          {activeFused.latencyUs.toLocaleString()} μs vs {eagerResult.latencyUs.toLocaleString()} μs
        </div>
      </div>

      {/* 2. VRAM Throughput Recovery */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
          <span>VRAM Throughput Recovery</span>
          <Database className="w-4 h-4 text-cyan-500" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-cyan-600 dark:text-cyan-400">
            {vramThroughputRecovery}%
          </span>
          <span className="text-xs text-slate-400 font-mono">bandwidth saved</span>
        </div>
        <div className="text-[11px] text-slate-500 mt-1 font-mono">
          Cut {dramTrafficSavedMB} MB DRAM traffic/pass
        </div>
      </div>

      {/* 3. DRAM Round-Trips Cut */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
          <span>DRAM Round-Trips</span>
          <RefreshCw className="w-4 h-4 text-indigo-500" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-indigo-600 dark:text-indigo-400">
            1 Pass
          </span>
          <span className="text-xs text-slate-400 font-mono">was 3 Passes</span>
        </div>
        <div className="text-[11px] text-slate-500 mt-1 font-mono">
          Intermediate tensors held in SRAM
        </div>
      </div>

      {/* 4. Warp Occupancy */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
          <span>Achieved Warp Occupancy</span>
          <Activity className="w-4 h-4 text-violet-500" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-violet-600 dark:text-violet-400">
            {activeFused.warpOccupancyPercent}%
          </span>
          <span className="text-xs text-slate-400 font-mono">vs {eagerResult.warpOccupancyPercent}% Eager</span>
        </div>
        <div className="text-[11px] text-slate-500 mt-1 font-mono">
          Nsight active warps / max peak
        </div>
      </div>

      {/* 5. Kernel Launches */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs col-span-2 lg:col-span-1">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
          <span>Kernel Dispatches</span>
          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-slate-900 dark:text-slate-100">
            1
          </span>
          <span className="text-xs text-slate-400 font-mono">single launch</span>
        </div>
        <div className="text-[11px] text-slate-500 mt-1 font-mono">
          Saved ~6.4 μs driver launch overhead
        </div>
      </div>
    </div>
  );
};
