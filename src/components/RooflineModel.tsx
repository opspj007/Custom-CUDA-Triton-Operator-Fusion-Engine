import React from 'react';
import { TrendingUp, Cpu, Info } from 'lucide-react';
import { ProfilingResult, KernelExecutionConfig, GPU_CATALOG } from '../core/gpu_simulation';

interface RooflineModelProps {
  config: KernelExecutionConfig;
  eagerResult: ProfilingResult;
  tritonResult: ProfilingResult;
  cudaResult: ProfilingResult;
  inductorResult: ProfilingResult;
}

export const RooflineModel: React.FC<RooflineModelProps> = ({
  config,
  eagerResult,
  tritonResult,
  cudaResult,
  inductorResult,
}) => {
  const gpu = GPU_CATALOG[config.gpuId];

  // Hardware boundaries
  const peakBandwidthGBs = gpu.memoryBandwidthGBs;
  const peakComputeTFlops = gpu.fp16PeakTFlops * (config.precision === 'FP32' ? 0.5 : 1.0);
  const ridgeIntensity = peakComputeTFlops / (peakBandwidthGBs / 1000); // FLOP/Byte

  // SVG dimensions
  const svgWidth = 720;
  const svgHeight = 360;
  const padLeft = 65;
  const padBottom = 50;
  const padRight = 30;
  const padTop = 30;

  // Log scale ranges:
  // X: Intensity from 0.1 to 1000 FLOP/Byte
  // Y: Performance from 0.05 to 1000 TFLOP/s
  const minX = 0.1;
  const maxX = 1000;
  const minY = 0.05;
  const maxY = Math.max(1200, peakComputeTFlops * 1.5);

  const scaleX = (val: number) => {
    const clamped = Math.max(minX, Math.min(maxX, val));
    const logVal = Math.log10(clamped);
    const logMin = Math.log10(minX);
    const logMax = Math.log10(maxX);
    return padLeft + ((logVal - logMin) / (logMax - logMin)) * (svgWidth - padLeft - padRight);
  };

  const scaleY = (val: number) => {
    const clamped = Math.max(minY, Math.min(maxY, val));
    const logVal = Math.log10(clamped);
    const logMin = Math.log10(minY);
    const logMax = Math.log10(maxY);
    return svgHeight - padBottom - ((logVal - logMin) / (logMax - logMin)) * (svgHeight - padTop - padBottom);
  };

  // Operating Points
  const calculateAchievedTflops = (res: ProfilingResult) => {
    return (res.arithmeticFlops / (res.latencyUs * 1e-6)) / 1e12;
  };

  const points = [
    {
      label: 'PyTorch Eager',
      intensity: eagerResult.arithmeticIntensityFlopPerByte,
      tflops: calculateAchievedTflops(eagerResult),
      color: '#f43f5e',
    },
    {
      label: 'TorchCompile / Inductor',
      intensity: inductorResult.arithmeticIntensityFlopPerByte,
      tflops: calculateAchievedTflops(inductorResult),
      color: '#3b82f6',
    },
    {
      label: 'Custom Triton Kernel',
      intensity: tritonResult.arithmeticIntensityFlopPerByte,
      tflops: calculateAchievedTflops(tritonResult),
      color: '#10b981',
    },
    {
      label: 'Custom CUDA C++',
      intensity: cudaResult.arithmeticIntensityFlopPerByte,
      tflops: calculateAchievedTflops(cudaResult),
      color: '#0d9488',
    },
  ];

  // Roofline polygon points:
  // Starts at (minX, minX * peakBandwidth)
  // Reaches ridge (ridgeIntensity, peakComputeTFlops)
  // Extends flat to (maxX, peakComputeTFlops)
  const xStart = minX;
  const yStart = (minX * peakBandwidthGBs) / 1000;
  const ptStart = `${scaleX(xStart)},${scaleY(yStart)}`;
  const ptRidge = `${scaleX(ridgeIntensity)},${scaleY(peakComputeTFlops)}`;
  const ptEnd = `${scaleX(maxX)},${scaleY(peakComputeTFlops)}`;

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800 gap-2">
        <div>
          <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-600" />
            GPU Empirical Roofline Model ({gpu.name})
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Operational points plotted against peak HBM memory bandwidth and peak tensor compute limits.
          </p>
        </div>
        <div className="text-xs font-mono text-slate-500">
          Ridge Point: <span className="font-bold text-slate-800 dark:text-slate-200">{ridgeIntensity.toFixed(1)} FLOP/Byte</span>
        </div>
      </div>

      {/* SVG Roofline Chart */}
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full min-w-[640px] text-xs font-mono select-none"
        >
          {/* Grid lines and Log Ticks */}
          {[0.1, 1, 10, 100, 1000].map((tick) => {
            const x = scaleX(tick);
            return (
              <g key={`x-${tick}`}>
                <line
                  x1={x}
                  y1={padTop}
                  x2={x}
                  y2={svgHeight - padBottom}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-slate-800 stroke-[1]"
                  strokeDasharray="3 3"
                />
                <text
                  x={x}
                  y={svgHeight - padBottom + 18}
                  textAnchor="middle"
                  className="fill-slate-400 text-[10px]"
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {[0.1, 1, 10, 100, 1000].map((tick) => {
            const y = scaleY(tick);
            return (
              <g key={`y-${tick}`}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={svgWidth - padRight}
                  y2={y}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-slate-800 stroke-[1]"
                  strokeDasharray="3 3"
                />
                <text
                  x={padLeft - 10}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-400 text-[10px]"
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {/* X and Y Axis Labels */}
          <text
            x={svgWidth / 2}
            y={svgHeight - 12}
            textAnchor="middle"
            className="fill-slate-500 font-sans text-xs font-medium"
          >
            Arithmetic Intensity (FLOPs / DRAM Byte) [Log Scale]
          </text>
          <text
            x={-svgHeight / 2}
            y={18}
            transform="rotate(-90)"
            textAnchor="middle"
            className="fill-slate-500 font-sans text-xs font-medium"
          >
            Attained Performance (TFLOP/s) [Log Scale]
          </text>

          {/* Roofline Ceiling Line */}
          <path
            d={`M ${ptStart} L ${ptRidge} L ${ptEnd}`}
            fill="none"
            stroke="#10b981"
            strokeWidth="2.5"
            strokeLinecap="round"
          />

          {/* Shaded Area under Roofline */}
          <path
            d={`M ${ptStart} L ${ptRidge} L ${ptEnd} L ${scaleX(maxX)},${scaleY(minY)} L ${scaleX(minX)},${scaleY(minY)} Z`}
            fill="url(#rooflineGradient)"
            opacity="0.1"
          />

          <defs>
            <linearGradient id="rooflineGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Ceiling Labels */}
          <text
            x={scaleX(1)}
            y={scaleY((1 * peakBandwidthGBs) / 1000) - 10}
            className="fill-emerald-600 dark:fill-emerald-400 text-[10px] font-bold"
            transform={`rotate(-28, ${scaleX(1)}, ${scaleY((1 * peakBandwidthGBs) / 1000) - 10})`}
          >
            Memory Bandwidth Bound ({peakBandwidthGBs} GB/s)
          </text>

          <text
            x={scaleX(300)}
            y={scaleY(peakComputeTFlops) - 8}
            className="fill-emerald-600 dark:fill-emerald-400 text-[10px] font-bold text-right"
            textAnchor="middle"
          >
            Compute Ceiling ({peakComputeTFlops} TFLOP/s)
          </text>

          {/* Operating Points with staggered clean callouts */}
          {points.map((pt, idx) => {
            const cx = scaleX(pt.intensity);
            const cy = scaleY(pt.tflops);
            // Stagger label y offsets to prevent overlap
            const yOffsets = [-14, 18, -24, 26];
            const dy = yOffsets[idx % yOffsets.length];
            return (
              <g key={pt.label} className="cursor-pointer">
                <circle cx={cx} cy={cy} r="5.5" fill={pt.color} stroke="#ffffff" strokeWidth="2" />
                {/* Connecting hairline */}
                <line
                  x1={cx}
                  y1={cy}
                  x2={cx + 14}
                  y2={cy + dy}
                  stroke={pt.color}
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />
                <rect
                  x={cx + 16}
                  y={cy + dy - 10}
                  width="180"
                  height="18"
                  rx="4"
                  className="fill-slate-900/90 dark:fill-slate-800/95"
                />
                <text
                  x={cx + 22}
                  y={cy + dy + 2}
                  fill="#ffffff"
                  className="font-bold text-[10px]"
                >
                  {pt.label}: {pt.tflops.toFixed(2)} TFLOPS
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Operating Points Comparison Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
        {points.map((pt) => (
          <div
            key={pt.label}
            className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40"
          >
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: pt.color }} />
              <span className="font-semibold text-slate-800 dark:text-slate-200 truncate">{pt.label}</span>
            </div>
            <div className="flex justify-between text-slate-500 mt-1">
              <span>Arithmetic Intensity:</span>
              <span className="font-bold text-slate-700 dark:text-slate-300">{pt.intensity} FLOP/B</span>
            </div>
            <div className="flex justify-between text-slate-500 mt-0.5">
              <span>Attained Compute:</span>
              <span className="font-bold text-emerald-600 dark:text-emerald-400">{pt.tflops.toFixed(2)} TFLOP/s</span>
            </div>
          </div>
        ))}
      </div>

      {/* Analysis description */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
        <div>
          <span className="font-bold text-slate-800 dark:text-slate-200">
            Why PyTorch Eager is Trapped on the Memory Slope:
          </span>
          <p className="text-slate-500 mt-1 leading-relaxed">
            Unfused eager execution achieves only <strong>{eagerResult.arithmeticIntensityFlopPerByte} FLOP/Byte</strong> arithmetic intensity because it performs 3 DRAM writes and 4 DRAM reads for basic elementwise operations. It operates far below the hardware ridge point (<strong>{ridgeIntensity.toFixed(1)} FLOP/Byte</strong>), causing GPU SM execution units to starve.
          </p>
        </div>
        <div>
          <span className="font-bold text-emerald-600 dark:text-emerald-400">
            Impact of Custom Kernel Operator Fusion:
          </span>
          <p className="text-slate-500 mt-1 leading-relaxed">
            By retaining intermediate activations in on-chip SRAM/registers, DRAM traffic drops from 7.2x to 3.1x. Arithmetic intensity jumps by <strong>2.3x</strong> to <strong>{tritonResult.arithmeticIntensityFlopPerByte} FLOP/Byte</strong>, driving attained performance from <strong>{calculateAchievedTflops(eagerResult).toFixed(2)} TFLOP/s</strong> to <strong>{calculateAchievedTflops(tritonResult).toFixed(2)} TFLOP/s</strong> (1.82x acceleration).
          </p>
        </div>
      </div>
    </div>
  );
};
