import React from 'react';
import { Sliders, Settings, Sparkles, Cpu } from 'lucide-react';
import { GPU_CATALOG, KernelExecutionConfig, PrecisionType } from '../core/gpu_simulation';

interface GpuConfigPanelProps {
  config: KernelExecutionConfig;
  onChangeConfig: (newConfig: Partial<KernelExecutionConfig>) => void;
  onApplyPreset: (presetName: string) => void;
}

export const GpuConfigPanel: React.FC<GpuConfigPanelProps> = ({
  config,
  onChangeConfig,
  onApplyPreset,
}) => {
  const presets = [
    { name: 'Ada SM 8.9 Benchmark (1.82x)', b: 16, s: 512, h: 2048, prec: 'FP16', gpu: 'rtx4090' },
    { name: 'Ampere A100 SXM (1.85x)', b: 16, s: 1024, h: 4096, prec: 'BF16', gpu: 'a100_80gb' },
    { name: 'LLaMA-3 8B Layer', b: 8, s: 2048, h: 4096, prec: 'FP16', gpu: 'rtx4090' },
    { name: 'BERT-Large Token', b: 32, s: 512, h: 1024, prec: 'FP16', gpu: 'rtx3090' },
    { name: 'DeepSeek-V2 MoE', b: 4, s: 4096, h: 5120, prec: 'BF16', gpu: 'h100_sxm' },
  ];

  const gpuOptions = Object.values(GPU_CATALOG);

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
      {/* Preset Row */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5 pb-4 border-b border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-slate-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Workload Presets
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {presets.map((preset) => (
            <button
              key={preset.name}
              onClick={() => {
                onChangeConfig({
                  gpuId: preset.gpu,
                  batchSize: preset.b,
                  seqLength: preset.s,
                  hiddenDim: preset.h,
                  precision: preset.prec as PrecisionType,
                });
                onApplyPreset(preset.name);
              }}
              className="px-2.5 py-1 text-xs font-medium rounded-md border border-slate-200 dark:border-slate-700 hover:border-emerald-500 dark:hover:border-emerald-500 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300 transition-colors whitespace-nowrap"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Grid of Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Hardware GPU */}
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1.5">
            Target GPU Architecture
          </label>
          <select
            value={config.gpuId}
            onChange={(e) => onChangeConfig({ gpuId: e.target.value })}
            className="w-full text-xs font-medium bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          >
            {gpuOptions.map((gpu) => (
              <option key={gpu.id} value={gpu.id}>
                {gpu.name} ({gpu.memoryBandwidthGBs} GB/s)
              </option>
            ))}
          </select>
          <div className="mt-1 text-[11px] text-slate-400 font-mono">
            {GPU_CATALOG[config.gpuId].smCount} SMs · {GPU_CATALOG[config.gpuId].sharedMemPerSmKB} KB Shmem/SM
          </div>
        </div>

        {/* Batch & Sequence Length */}
        <div>
          <div className="flex justify-between items-center mb-1.5">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400">
              Batch Size (B) &amp; Seq Length (S)
            </label>
            <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
              {(config.batchSize * config.seqLength).toLocaleString()} tokens
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-[10px] text-slate-400 uppercase">Batch</span>
              <select
                value={config.batchSize}
                onChange={(e) => onChangeConfig({ batchSize: Number(e.target.value) })}
                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100"
              >
                {[1, 4, 8, 16, 32, 64].map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <span className="text-[10px] text-slate-400 uppercase">Seq Len</span>
              <select
                value={config.seqLength}
                onChange={(e) => onChangeConfig({ seqLength: Number(e.target.value) })}
                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100"
              >
                {[128, 256, 512, 1024, 2048, 4096].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Hidden Dimension & Precision */}
        <div>
          <div className="flex justify-between items-center mb-1.5">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400">
              Hidden Dim (D) &amp; Dtype
            </label>
            <span className="text-[11px] font-mono text-slate-500">
              {config.hiddenDim} dim · {config.precision}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-[10px] text-slate-400 uppercase">Hidden Dim</span>
              <select
                value={config.hiddenDim}
                onChange={(e) => onChangeConfig({ hiddenDim: Number(e.target.value) })}
                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100"
              >
                {[768, 1024, 2048, 4096, 5120, 8192].map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <span className="text-[10px] text-slate-400 uppercase">Precision</span>
              <select
                value={config.precision}
                onChange={(e) => onChangeConfig({ precision: e.target.value as PrecisionType })}
                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100"
              >
                {['FP16', 'BF16', 'FP32'].map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Triton Tunables */}
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1.5">
            Triton Kernel Autotuner
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-[10px] text-slate-400 uppercase">BLOCK_SIZE</span>
              <select
                value={config.tritonBlockSize}
                onChange={(e) => onChangeConfig({ tritonBlockSize: Number(e.target.value) })}
                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100"
              >
                {[128, 256, 512, 1024, 2048].map((bs) => (
                  <option key={bs} value={bs}>
                    {bs}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <span className="text-[10px] text-slate-400 uppercase">Warps</span>
              <select
                value={config.numWarps}
                onChange={(e) => onChangeConfig({ numWarps: Number(e.target.value) })}
                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100"
              >
                {[2, 4, 8, 16].map((w) => (
                  <option key={w} value={w}>
                    {w} warps
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
