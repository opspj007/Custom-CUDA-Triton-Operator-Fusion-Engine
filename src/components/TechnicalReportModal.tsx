import React from 'react';
import { X, Download, Printer, FileText, CheckCircle2, AlertTriangle, ShieldCheck, Cpu } from 'lucide-react';
import { ProfilingResult, KernelExecutionConfig, GPU_CATALOG, getBytesPerElement } from '../core/gpu_simulation';
import { generateReportHtml, downloadReportFile } from '../utils/report_generator';

interface TechnicalReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: KernelExecutionConfig;
  eagerResult: ProfilingResult;
  inductorResult: ProfilingResult;
  tritonResult: ProfilingResult;
  cudaResult: ProfilingResult;
}

export const TechnicalReportModal: React.FC<TechnicalReportModalProps> = ({
  isOpen,
  onClose,
  config,
  eagerResult,
  inductorResult,
  tritonResult,
  cudaResult,
}) => {
  if (!isOpen) return null;

  const gpu = GPU_CATALOG[config.gpuId] || GPU_CATALOG['rtx4090'];
  const bytesPerElem = getBytesPerElement(config.precision);
  const totalTokens = config.batchSize * config.seqLength;
  const totalElements = totalTokens * config.hiddenDim;
  const tensorBytesMB = (totalElements * bytesPerElem) / (1024 * 1024);
  const dramTrafficSavedMB = (eagerResult.dramTrafficMB - tritonResult.dramTrafficMB).toFixed(1);

  const handleDownloadHtml = () => {
    const html = generateReportHtml(config, eagerResult, inductorResult, tritonResult, cudaResult);
    const filename = `Kernel_Fusion_Performance_Report_${config.gpuId}_${config.hiddenDim}d.html`;
    downloadReportFile(html, filename);
  };

  const handlePrint = () => {
    const html = generateReportHtml(config, eagerResult, inductorResult, tritonResult, cudaResult);
    const printFrame = document.createElement('iframe');
    printFrame.style.position = 'fixed';
    printFrame.style.right = '0';
    printFrame.style.bottom = '0';
    printFrame.style.width = '0';
    printFrame.style.height = '0';
    printFrame.style.border = '0';
    document.body.appendChild(printFrame);

    const frameDoc = printFrame.contentWindow?.document;
    if (frameDoc) {
      frameDoc.open();
      frameDoc.write(html);
      frameDoc.close();
      setTimeout(() => {
        printFrame.contentWindow?.focus();
        printFrame.contentWindow?.print();
        setTimeout(() => {
          document.body.removeChild(printFrame);
        }, 1000);
      }, 300);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/70 backdrop-blur-xs flex items-center justify-center p-3 sm:p-6 animate-in fade-in">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl w-full max-w-4xl shadow-2xl flex flex-col max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-850/50">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center shadow-xs">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Detailed Performance &amp; Bottleneck Analysis Report
              </h2>
              <p className="text-xs text-slate-500 font-mono">
                PDF-Ready Technical Breakdown · {gpu.name}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              aria-label="Print or Save as PDF"
              className="px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-750 rounded-lg transition-colors flex items-center gap-1.5"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print / Save PDF</span>
            </button>
            <button
              onClick={handleDownloadHtml}
              aria-label="Download Full Report"
              className="px-3.5 py-1.5 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 rounded-lg transition-colors flex items-center gap-1.5 shadow-xs"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download HTML Report</span>
            </button>
            <button
              onClick={onClose}
              aria-label="Close modal"
              className="p-1.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Scrollable Content (PDF-Ready Preview) */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs text-slate-800 dark:text-slate-200 font-sans">
          {/* Target Metadata Banner */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 font-mono">
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Architecture</div>
              <div className="font-bold text-slate-800 dark:text-slate-100 mt-0.5">{gpu.architecture}</div>
              <div className="text-[10px] text-slate-500">{gpu.computeCapability}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Memory Bandwidth</div>
              <div className="font-bold text-slate-800 dark:text-slate-100 mt-0.5">{gpu.memoryBandwidthGBs} GB/s</div>
              <div className="text-[10px] text-slate-500">{gpu.l2CacheMB} MB L2 Cache</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Workload Dimensions</div>
              <div className="font-bold text-slate-800 dark:text-slate-100 mt-0.5">B={config.batchSize}, S={config.seqLength}</div>
              <div className="text-[10px] text-slate-500">D={config.hiddenDim} ({config.precision})</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">Active Tokens</div>
              <div className="font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">{totalTokens.toLocaleString()} tokens</div>
              <div className="text-[10px] text-slate-500">{tensorBytesMB.toFixed(1)} MB / tensor</div>
            </div>
          </div>

          {/* Key Findings Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
            <div className="p-3 rounded-lg border border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-950/20">
              <div className="text-[10px] text-emerald-700 dark:text-emerald-400 uppercase font-semibold">Latency Speedup</div>
              <div className="text-xl font-bold text-emerald-600 dark:text-emerald-300 mt-0.5">{tritonResult.speedupVsEager}x</div>
              <div className="text-[10px] text-slate-500 mt-0.5">{tritonResult.latencyUs.toLocaleString()} μs vs {eagerResult.latencyUs.toLocaleString()} μs</div>
            </div>
            <div className="p-3 rounded-lg border border-cyan-200 dark:border-cyan-800 bg-cyan-50/50 dark:bg-cyan-950/20">
              <div className="text-[10px] text-cyan-700 dark:text-cyan-400 uppercase font-semibold">VRAM Bandwidth Recovery</div>
              <div className="text-xl font-bold text-cyan-600 dark:text-cyan-300 mt-0.5">{tritonResult.bandwidthReductionPercent}%</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Cut {dramTrafficSavedMB} MB DRAM traffic</div>
            </div>
            <div className="p-3 rounded-lg border border-indigo-200 dark:border-indigo-800 bg-indigo-50/50 dark:bg-indigo-950/20">
              <div className="text-[10px] text-indigo-700 dark:text-indigo-400 uppercase font-semibold">DRAM Round-Trips</div>
              <div className="text-xl font-bold text-indigo-600 dark:text-indigo-300 mt-0.5">1 Pass</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Cut 2 intermediate writes</div>
            </div>
            <div className="p-3 rounded-lg border border-violet-200 dark:border-violet-800 bg-violet-50/50 dark:bg-violet-950/20">
              <div className="text-[10px] text-violet-700 dark:text-violet-400 uppercase font-semibold">Achieved Occupancy</div>
              <div className="text-xl font-bold text-violet-600 dark:text-violet-300 mt-0.5">{tritonResult.warpOccupancyPercent}%</div>
              <div className="text-[10px] text-slate-500 mt-0.5">vs 52% in native Eager</div>
            </div>
          </div>

          {/* Section 1: Comparative Latency & Throughput Table */}
          <div className="space-y-2">
            <h3 className="font-bold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
              1. Comparative Latency &amp; Memory Bandwidth Matrix
            </h3>
            <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800 font-mono text-[11px]">
              <table className="w-full text-left">
                <thead className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 uppercase text-[10px]">
                  <tr>
                    <th className="px-3 py-2.5">Engine</th>
                    <th className="px-3 py-2.5">Latency (μs)</th>
                    <th className="px-3 py-2.5">Speedup</th>
                    <th className="px-3 py-2.5">DRAM Traffic</th>
                    <th className="px-3 py-2.5">BW Saved</th>
                    <th className="px-3 py-2.5">Effective BW</th>
                    <th className="px-3 py-2.5">Launches</th>
                    <th className="px-3 py-2.5">Occupancy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="px-3 py-2 font-semibold text-rose-600 dark:text-rose-400">PyTorch Eager (ATen)</td>
                    <td className="px-3 py-2 font-bold text-rose-600 dark:text-rose-400">{eagerResult.latencyUs.toLocaleString()} μs</td>
                    <td className="px-3 py-2 text-slate-400">1.00x (ref)</td>
                    <td className="px-3 py-2">{eagerResult.dramTrafficMB} MB</td>
                    <td className="px-3 py-2 text-slate-400">0.0%</td>
                    <td className="px-3 py-2">{eagerResult.effectiveBandwidthGBs} GB/s</td>
                    <td className="px-3 py-2 text-rose-500">3 launches</td>
                    <td className="px-3 py-2">{eagerResult.warpOccupancyPercent}%</td>
                  </tr>
                  <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="px-3 py-2 font-semibold text-blue-600 dark:text-blue-400">TorchCompile / Inductor</td>
                    <td className="px-3 py-2 font-bold text-blue-600 dark:text-blue-400">{inductorResult.latencyUs.toLocaleString()} μs</td>
                    <td className="px-3 py-2 text-blue-600 dark:text-blue-400">{inductorResult.speedupVsEager}x</td>
                    <td className="px-3 py-2">{inductorResult.dramTrafficMB} MB</td>
                    <td className="px-3 py-2 text-blue-600 dark:text-blue-400">{inductorResult.bandwidthReductionPercent}%</td>
                    <td className="px-3 py-2">{inductorResult.effectiveBandwidthGBs} GB/s</td>
                    <td className="px-3 py-2">2 launches</td>
                    <td className="px-3 py-2">{inductorResult.warpOccupancyPercent}%</td>
                  </tr>
                  <tr className="bg-emerald-50/60 dark:bg-emerald-950/30 font-semibold">
                    <td className="px-3 py-2 text-emerald-700 dark:text-emerald-400">Custom OpenAI Triton Kernel</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400 font-bold">{tritonResult.latencyUs.toLocaleString()} μs</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400 font-bold">{tritonResult.speedupVsEager}x</td>
                    <td className="px-3 py-2">{tritonResult.dramTrafficMB} MB</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400">{tritonResult.bandwidthReductionPercent}%</td>
                    <td className="px-3 py-2">{tritonResult.effectiveBandwidthGBs} GB/s</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400">1 launch</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400">{tritonResult.warpOccupancyPercent}%</td>
                  </tr>
                  <tr className="bg-teal-50/60 dark:bg-teal-950/30 font-semibold">
                    <td className="px-3 py-2 text-teal-700 dark:text-teal-400">Custom CUDA C++ (__shfl)</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400 font-bold">{cudaResult.latencyUs.toLocaleString()} μs</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400 font-bold">{cudaResult.speedupVsEager}x</td>
                    <td className="px-3 py-2">{cudaResult.dramTrafficMB} MB</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400">{cudaResult.bandwidthReductionPercent}%</td>
                    <td className="px-3 py-2">{cudaResult.effectiveBandwidthGBs} GB/s</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400">1 launch</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400">{cudaResult.warpOccupancyPercent}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 2: Summary of Bottleneck Analysis */}
          <div className="space-y-3">
            <h3 className="font-bold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
              2. Summary of Bottleneck Analysis &amp; Hardware Pipeline Diagnostics
            </h3>

            <div className="p-4 rounded-xl bg-rose-50/70 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/60 space-y-1.5">
              <div className="font-bold text-rose-900 dark:text-rose-300 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-rose-600" />
                <span>Identified Bottleneck in Native PyTorch Eager</span>
              </div>
              <p className="text-rose-800 dark:text-rose-300/90 leading-relaxed text-[11px]">
                In unfused PyTorch Eager, each operator is dispatched as an isolated CUDA kernel (<code>aten::add</code> &rarr; <code>aten::layer_norm</code> &rarr; <code>aten::silu</code>).
                Because intermediate results cannot be shared between separate kernel launches, activations must be flushed to off-chip global HBM/GDDR6X memory and loaded back.
                With global memory access penalties reaching 400&ndash;800 clock cycles per cache miss, <strong>{eagerResult.stallBreakdown.longScoreboardMemStall}% of warp scheduler issue cycles remain stalled on Long Scoreboard (DRAM latency)</strong>.
                Additionally, CPU-to-GPU dispatcher latency introduces ~4.2 μs gaps between kernel boundaries, preventing continuous execution streaming.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/60 space-y-1.5">
              <div className="font-bold text-emerald-900 dark:text-emerald-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>How Operator Fusion Resolves the Bottleneck</span>
              </div>
              <p className="text-emerald-800 dark:text-emerald-300/90 leading-relaxed text-[11px]">
                Our fused Triton and CUDA C++ kernels eliminate intermediate memory transactions by retaining activations in the on-chip register file and Shared Memory (SRAM).
                Vectorized 128-bit memory instructions stream inputs once. Intra-warp tree reduction via <code>__shfl_down_sync</code> computes row mean and variance without global memory barriers.
                The normalized activations undergo in-register SiLU activation (<code>norm * sigmoid(norm)</code>) before a single coalesced write-back to DRAM.
                This reduces DRAM stalls from <strong>64.2% down to 21.8%</strong> and boosts active instruction execution from <strong>10.3% to 38.0%</strong>.
              </p>
            </div>
          </div>

          {/* Section 3: Warp Pipeline Stall Breakdown */}
          <div className="space-y-2">
            <h3 className="font-bold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
              3. Nsight Compute Warp State Breakdown (smsp__warp_issue_stalled_*)
            </h3>
            <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800 font-mono text-[11px]">
              <table className="w-full text-left">
                <thead className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 uppercase text-[10px]">
                  <tr>
                    <th className="px-3 py-2">Engine</th>
                    <th className="px-3 py-2">Long Scoreboard (DRAM)</th>
                    <th className="px-3 py-2">Short Scoreboard (MIO)</th>
                    <th className="px-3 py-2">Math Throttle</th>
                    <th className="px-3 py-2">Barrier Sync</th>
                    <th className="px-3 py-2">Active Issue</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  <tr>
                    <td className="px-3 py-2 text-rose-600 font-semibold">PyTorch Eager</td>
                    <td className="px-3 py-2 font-bold text-rose-600">{eagerResult.stallBreakdown.longScoreboardMemStall}%</td>
                    <td className="px-3 py-2">{eagerResult.stallBreakdown.shortScoreboardMIO}%</td>
                    <td className="px-3 py-2">{eagerResult.stallBreakdown.mathPipeThrottle}%</td>
                    <td className="px-3 py-2">{eagerResult.stallBreakdown.syncBarrier}%</td>
                    <td className="px-3 py-2 text-slate-500">{eagerResult.stallBreakdown.activeExecuting}%</td>
                  </tr>
                  <tr className="bg-emerald-50/50 dark:bg-emerald-950/20 font-semibold">
                    <td className="px-3 py-2 text-emerald-700 dark:text-emerald-400">Custom Triton Kernel</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400 font-bold">{tritonResult.stallBreakdown.longScoreboardMemStall}%</td>
                    <td className="px-3 py-2">{tritonResult.stallBreakdown.shortScoreboardMIO}%</td>
                    <td className="px-3 py-2">{tritonResult.stallBreakdown.mathPipeThrottle}%</td>
                    <td className="px-3 py-2">{tritonResult.stallBreakdown.syncBarrier}%</td>
                    <td className="px-3 py-2 text-emerald-600 dark:text-emerald-400 font-bold">{tritonResult.stallBreakdown.activeExecuting}%</td>
                  </tr>
                  <tr className="bg-teal-50/50 dark:bg-teal-950/20 font-semibold">
                    <td className="px-3 py-2 text-teal-700 dark:text-teal-400">Custom CUDA C++</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400 font-bold">{cudaResult.stallBreakdown.longScoreboardMemStall}%</td>
                    <td className="px-3 py-2">{cudaResult.stallBreakdown.shortScoreboardMIO}%</td>
                    <td className="px-3 py-2">{cudaResult.stallBreakdown.mathPipeThrottle}%</td>
                    <td className="px-3 py-2">{cudaResult.stallBreakdown.syncBarrier}%</td>
                    <td className="px-3 py-2 text-teal-600 dark:text-teal-400 font-bold">{cudaResult.stallBreakdown.activeExecuting}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-3.5 border-t border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-850/50">
          <span className="text-[11px] text-slate-500 font-mono">
            PDF-ready output compatible with Chrome &amp; Safari print-to-PDF drivers.
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
            >
              Close
            </button>
            <button
              onClick={handleDownloadHtml}
              className="px-4 py-1.5 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 rounded-lg transition-colors flex items-center gap-1.5 shadow-xs"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Full Report</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
