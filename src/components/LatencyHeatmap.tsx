import React, { useState, useMemo } from 'react';
import {
  Grid,
  Zap,
  Target,
  Sparkles,
  Layers,
  ArrowRight,
  HardDrive,
  Cpu,
  Activity,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { KernelExecutionConfig, GPU_CATALOG } from '../core/gpu_simulation';
import {
  generateShapeHeatmapGrid,
  getHeatmapCellColor,
  HeatmapCell,
  HeatmapMetricMode,
  HEATMAP_BATCH_SIZES,
  HEATMAP_SEQ_LENGTHS,
} from '../core/heatmap_logic';

interface LatencyHeatmapProps {
  config: KernelExecutionConfig;
  onApplyShape?: (batchSize: number, seqLength: number) => void;
}

export const LatencyHeatmap: React.FC<LatencyHeatmapProps> = ({ config, onApplyShape }) => {
  const [metricMode, setMetricMode] = useState<HeatmapMetricMode>('tritonSpeedup');
  const [hoveredCell, setHoveredCell] = useState<HeatmapCell | null>(null);
  const [selectedCell, setSelectedCell] = useState<HeatmapCell | null>(null);

  const matrix = useMemo(() => {
    return generateShapeHeatmapGrid({
      gpuId: config.gpuId,
      hiddenDim: config.hiddenDim,
      precision: config.precision,
      tritonBlockSize: config.tritonBlockSize,
      numWarps: config.numWarps,
    });
  }, [config.gpuId, config.hiddenDim, config.precision, config.tritonBlockSize, config.numWarps]);

  // Set default selected cell to current active shape or sweet spot
  const activeCell = useMemo(() => {
    if (selectedCell) return selectedCell;
    if (hoveredCell) return hoveredCell;
    // Find matching active config cell or sweet spot
    for (const row of matrix.cells) {
      for (const c of row) {
        if (c.batchSize === config.batchSize && c.seqLength === config.seqLength) {
          return c;
        }
      }
    }
    return matrix.sweetSpotCell;
  }, [selectedCell, hoveredCell, matrix, config.batchSize, config.seqLength]);

  const sweetSpot = matrix.sweetSpotCell;
  const isCurrentConfigSweetSpot =
    config.batchSize === sweetSpot.batchSize && config.seqLength === sweetSpot.seqLength;

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-6">
      {/* Top Header Strip */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center font-bold">
              <Grid className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 tracking-tight flex items-center gap-2">
                <span>Tensor Shape Performance Heatmap</span>
                <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20">
                  Batch Size (B) × Seq Length (S)
                </span>
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Evaluates speedup, latency, and SM wave quantization across 49 shape permutations on {matrix.gpu.name}.
              </p>
            </div>
          </div>
        </div>

        {/* Metric Selector Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <label className="text-[11px] font-mono uppercase text-slate-400 mr-1">
            Display Metric:
          </label>
          <div className="inline-flex bg-slate-100 dark:bg-slate-800/80 p-1 rounded-lg text-xs font-medium">
            <button
              onClick={() => setMetricMode('tritonSpeedup')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                metricMode === 'tritonSpeedup'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              Triton Speedup (×)
            </button>
            <button
              onClick={() => setMetricMode('cudaSpeedup')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                metricMode === 'cudaSpeedup'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              CUDA Speedup (×)
            </button>
            <button
              onClick={() => setMetricMode('tritonLatency')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                metricMode === 'tritonLatency'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              Fused Latency (μs)
            </button>
            <button
              onClick={() => setMetricMode('eagerLatency')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                metricMode === 'eagerLatency'
                  ? 'bg-rose-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              Eager Latency (μs)
            </button>
            <button
              onClick={() => setMetricMode('dramSaved')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                metricMode === 'dramSaved'
                  ? 'bg-purple-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
              }`}
            >
              DRAM Saved (MB)
            </button>
          </div>
        </div>
      </div>

      {/* Sweet Spot Highlight Banner */}
      <div className="bg-gradient-to-r from-emerald-50 via-slate-50 to-emerald-50 dark:from-emerald-950/40 dark:via-slate-900 dark:to-emerald-950/40 border border-emerald-500/30 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 mt-0.5">
            <Target className="w-5 h-5" />
          </div>
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-emerald-700 dark:text-emerald-300 uppercase tracking-wider">
                Optimal Fusion Sweet Spot Identified
              </span>
              <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-emerald-600 text-white">
                B={sweetSpot.batchSize} × S={sweetSpot.seqLength} ({sweetSpot.tokens.toLocaleString()} tokens)
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-300">
              {sweetSpot.reason}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {onApplyShape && (
            <button
              onClick={() => onApplyShape(sweetSpot.batchSize, sweetSpot.seqLength)}
              disabled={isCurrentConfigSweetSpot}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors flex items-center gap-1.5 shadow-xs whitespace-nowrap ${
                isCurrentConfigSweetSpot
                  ? 'bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-default'
                  : 'bg-emerald-600 hover:bg-emerald-700 text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>{isCurrentConfigSweetSpot ? 'Active in Studio' : 'Apply Sweet Spot Shape'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Grid & Inspector Split Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap Grid (7 cols) */}
        <div className="lg:col-span-7 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span className="font-mono text-[11px] uppercase tracking-wider">
              Matrix: Batch (Rows) vs Sequence (Columns)
            </span>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/20 border border-emerald-500/40 inline-block" />
                Lower
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm bg-emerald-600 inline-block" />
                Optimal Sweet Spot
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm border-2 border-amber-400 inline-block" />
                Current Config
              </span>
            </div>
          </div>

          <div className="overflow-x-auto pb-2">
            <div className="min-w-[500px]">
              {/* Columns Header (Sequence Lengths) */}
              <div className="grid grid-cols-8 gap-1.5 mb-1.5">
                <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center justify-center font-bold">
                  B \ S
                </div>
                {HEATMAP_SEQ_LENGTHS.map((s) => (
                  <div
                    key={s}
                    className="text-[11px] font-mono text-center text-slate-600 dark:text-slate-400 font-semibold py-1 bg-slate-50 dark:bg-slate-800/40 rounded"
                  >
                    {s}
                  </div>
                ))}
              </div>

              {/* Rows (Batch Sizes) */}
              {matrix.cells.map((row, bIdx) => {
                const b = matrix.batchSizes[bIdx];
                return (
                  <div key={b} className="grid grid-cols-8 gap-1.5 mb-1.5">
                    {/* Row Header */}
                    <div className="text-[11px] font-mono text-slate-600 dark:text-slate-400 font-semibold flex items-center justify-center bg-slate-50 dark:bg-slate-800/40 rounded">
                      B={b}
                    </div>

                    {/* Heatmap Cells */}
                    {row.map((cell) => {
                      const isCurrent =
                        cell.batchSize === config.batchSize && cell.seqLength === config.seqLength;
                      const isSelected =
                        activeCell.batchSize === cell.batchSize &&
                        activeCell.seqLength === cell.seqLength;
                      const isSweet = cell.isSweetSpot;
                      const styling = getHeatmapCellColor(cell, metricMode, matrix);

                      return (
                        <button
                          key={`${cell.batchSize}-${cell.seqLength}`}
                          onClick={() => setSelectedCell(cell)}
                          onMouseEnter={() => setHoveredCell(cell)}
                          onMouseLeave={() => setHoveredCell(null)}
                          style={{
                            backgroundColor: styling.backgroundColor,
                            color: styling.textColor,
                          }}
                          className={`h-11 rounded-lg text-xs font-mono font-bold transition-all transform flex flex-col items-center justify-center relative cursor-pointer ${
                            isSweet
                              ? 'ring-2 ring-emerald-400 dark:ring-emerald-300 shadow-md'
                              : isCurrent
                              ? 'ring-2 ring-amber-400 shadow-xs'
                              : isSelected
                              ? 'ring-2 ring-slate-400'
                              : 'hover:scale-[1.04]'
                          }`}
                        >
                          {/* Sweet spot indicator badge */}
                          {isSweet && (
                            <span className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-500 rounded-full flex items-center justify-center text-[8px] text-white font-bold ring-1 ring-white">
                              ★
                            </span>
                          )}

                          <span className="text-[11px]">{styling.displayValue}</span>
                          <span className="text-[9px] opacity-75 font-normal">
                            {cell.tokens < 1000
                              ? `${cell.tokens}t`
                              : `${(cell.tokens / 1000).toFixed(0)}k`}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Selected Cell Inspector & Comparison Card (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-700">
              <div>
                <span className="text-[10px] font-mono uppercase text-slate-400 block">
                  Cell Shape Deep-Dive
                </span>
                <h4 className="text-sm font-bold font-mono text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <span>
                    B={activeCell.batchSize} × S={activeCell.seqLength} × D={matrix.hiddenDim}
                  </span>
                  {activeCell.isSweetSpot && (
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-600 text-white">
                      SWEET SPOT
                    </span>
                  )}
                </h4>
              </div>
              <span className="text-xs font-mono text-slate-500">
                {activeCell.tokens.toLocaleString()} tokens
              </span>
            </div>

            {/* Shape Performance Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700 text-slate-400 text-[10px]">
                    <th className="pb-1.5 font-medium">ENGINE</th>
                    <th className="pb-1.5 font-medium text-right">LATENCY</th>
                    <th className="pb-1.5 font-medium text-right">SPEEDUP</th>
                    <th className="pb-1.5 font-medium text-right">DRAM</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200/60 dark:divide-slate-700/60">
                  <tr>
                    <td className="py-2 text-rose-600 dark:text-rose-400 font-semibold">PyTorch Eager</td>
                    <td className="py-2 text-right">{activeCell.eagerLatencyUs} μs</td>
                    <td className="py-2 text-right text-slate-400">1.00× (ref)</td>
                    <td className="py-2 text-right">{activeCell.eagerDramMB} MB</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-amber-600 dark:text-amber-400">Unfused CUDA</td>
                    <td className="py-2 text-right">{activeCell.unfusedCudaLatencyUs} μs</td>
                    <td className="py-2 text-right text-amber-500">
                      {(activeCell.eagerLatencyUs / activeCell.unfusedCudaLatencyUs).toFixed(2)}×
                    </td>
                    <td className="py-2 text-right">{activeCell.eagerDramMB} MB</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-blue-600 dark:text-blue-400">TorchInductor</td>
                    <td className="py-2 text-right">{activeCell.inductorLatencyUs} μs</td>
                    <td className="py-2 text-right text-blue-500">{activeCell.inductorSpeedup.toFixed(2)}×</td>
                    <td className="py-2 text-right">{Math.round(activeCell.eagerDramMB * 0.64)} MB</td>
                  </tr>
                  <tr className="bg-emerald-500/10 font-bold">
                    <td className="py-2 text-emerald-600 dark:text-emerald-400">Custom Triton</td>
                    <td className="py-2 text-right text-emerald-600 dark:text-emerald-400">
                      {activeCell.tritonLatencyUs} μs
                    </td>
                    <td className="py-2 text-right text-emerald-600 dark:text-emerald-400">
                      {activeCell.tritonSpeedup.toFixed(2)}×
                    </td>
                    <td className="py-2 text-right text-emerald-600 dark:text-emerald-400">
                      {activeCell.fusedDramMB} MB
                    </td>
                  </tr>
                  <tr className="bg-emerald-500/20 font-bold">
                    <td className="py-2 text-emerald-700 dark:text-emerald-300">Custom CUDA C++</td>
                    <td className="py-2 text-right text-emerald-700 dark:text-emerald-300">
                      {activeCell.cudaLatencyUs} μs
                    </td>
                    <td className="py-2 text-right text-emerald-700 dark:text-emerald-300">
                      {activeCell.cudaSpeedup.toFixed(2)}×
                    </td>
                    <td className="py-2 text-right text-emerald-700 dark:text-emerald-300">
                      {activeCell.fusedDramMB} MB
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Wave Quantization Meter */}
            <div className="space-y-1.5 pt-2 border-t border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500 font-mono text-[11px]">SM Wave Efficiency</span>
                <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
                  {activeCell.waveEfficiencyPercent}% on {matrix.gpu.smCount} SMs
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    activeCell.waveEfficiencyPercent >= 85
                      ? 'bg-emerald-500'
                      : activeCell.waveEfficiencyPercent >= 60
                      ? 'bg-amber-500'
                      : 'bg-rose-500'
                  }`}
                  style={{ width: `${activeCell.waveEfficiencyPercent}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                {activeCell.tokens < 2048
                  ? 'Under-saturates GPU SMs. Fractional tail waves create idle compute cycles.'
                  : activeCell.waveEfficiencyPercent >= 90
                  ? 'Optimal integer wave multiple across all Streaming Multiprocessors with minimal tail wave idle time.'
                  : 'Full GPU saturation achieved with minor tail wave quantization penalty.'}
              </p>
            </div>

            {/* Action button to load this shape into studio */}
            {onApplyShape && (
              <button
                onClick={() => onApplyShape(activeCell.batchSize, activeCell.seqLength)}
                disabled={
                  config.batchSize === activeCell.batchSize && config.seqLength === activeCell.seqLength
                }
                className="w-full py-2 px-3 text-xs font-medium rounded-lg bg-slate-200 dark:bg-slate-700 hover:bg-emerald-600 hover:text-white dark:hover:bg-emerald-600 transition-colors flex items-center justify-center gap-1.5 font-mono disabled:opacity-50"
              >
                <ArrowRight className="w-3.5 h-3.5" />
                <span>
                  {config.batchSize === activeCell.batchSize && config.seqLength === activeCell.seqLength
                    ? 'Current Studio Shape'
                    : `Load Shape (B=${activeCell.batchSize}, S=${activeCell.seqLength}) into Studio`}
                </span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 3 Performance Regimes Guide Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs border-t border-slate-100 dark:border-slate-800">
        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800 space-y-1.5">
          <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
            Regime 1: Small Shapes (&lt;1K Tokens)
          </div>
          <p className="text-slate-500 text-[11px] leading-relaxed">
            Dominated by CPU driver launch bubbles (~9.8μs in eager vs 3.4μs in Triton). Speedup is modest (1.3×–1.6×) because total kernel runtime is too short to saturate memory buses.
          </p>
        </div>

        <div className="p-3 rounded-lg bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-500/30 space-y-1.5">
          <div className="font-semibold text-emerald-900 dark:text-emerald-200 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
            Regime 2: Fusion Golden Zone (4K–32K Tokens)
          </div>
          <p className="text-slate-600 dark:text-slate-300 text-[11px] leading-relaxed">
            The architectural sweet spot: full SM grid occupancy, zero launch overhead penalty, and complete DRAM traffic elimination yielding peak <strong>2.6×–2.85× speedup</strong>.
          </p>
        </div>

        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800 space-y-1.5">
          <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
            Regime 3: Ultra-Large (&gt;64K Tokens)
          </div>
          <p className="text-slate-500 text-[11px] leading-relaxed">
            Pure DRAM bandwidth bound. The GPU memory controller is 100% saturated. Fused kernels consistently deliver a clean ~1.82× speedup strictly reflecting the 7.2× to 3.1× byte reduction.
          </p>
        </div>
      </div>
    </div>
  );
};
