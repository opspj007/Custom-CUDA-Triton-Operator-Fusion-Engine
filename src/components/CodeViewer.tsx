import React, { useState } from 'react';
import { Copy, Check, Code, FileText, Split, Terminal } from 'lucide-react';
import { KERNEL_CODE_FILES, KernelCodeSnippet } from '../data/kernel_code';

export const CodeViewer: React.FC = () => {
  const [selectedFileId, setSelectedFileId] = useState<string>('triton');
  const [copied, setCopied] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'source' | 'diff'>('source');

  const activeSnippet: KernelCodeSnippet = KERNEL_CODE_FILES[selectedFileId] || KERNEL_CODE_FILES.triton;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeSnippet.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
      {/* Top File Selector and Action Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800 gap-3">
        {/* File tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto">
          {Object.values(KERNEL_CODE_FILES).map((snippet) => (
            <button
              key={snippet.id}
              onClick={() => {
                setSelectedFileId(snippet.id);
                setViewMode('source');
              }}
              className={`px-3 py-1.5 text-xs font-mono font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 ${
                selectedFileId === snippet.id
                  ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 shadow-xs'
                  : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>{snippet.filename}</span>
            </button>
          ))}
        </div>

        {/* View Mode & Copy Button */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center bg-slate-100 dark:bg-slate-800 p-0.5 rounded-lg text-xs font-medium">
            <button
              onClick={() => setViewMode('source')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                viewMode === 'source' ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs' : 'text-slate-500'
              }`}
            >
              Source
            </button>
            <button
              onClick={() => setViewMode('diff')}
              className={`px-2.5 py-1 rounded-md transition-colors flex items-center gap-1 ${
                viewMode === 'diff' ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs' : 'text-slate-500'
              }`}
            >
              <Split className="w-3 h-3" />
              Diff (Eager vs Fused)
            </button>
          </div>

          <button
            onClick={handleCopy}
            className="px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-1.5"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy Code'}</span>
          </button>
        </div>
      </div>

      {/* Description & Key Optimizations */}
      <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800 text-xs">
        <div className="font-semibold text-slate-900 dark:text-slate-100 mb-1">
          {activeSnippet.name}
        </div>
        <p className="text-slate-600 dark:text-slate-400 mb-2">
          {activeSnippet.description}
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px] font-mono">
          {activeSnippet.highlights.map((h, i) => (
            <div key={i} className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-750 text-slate-700 dark:text-slate-300">
              ✓ {h}
            </div>
          ))}
        </div>
      </div>

      {/* Code Display Area */}
      {viewMode === 'source' ? (
        <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-950 text-slate-200">
          <div className="flex items-center justify-between px-4 py-2 bg-slate-900/80 border-b border-slate-800 text-[11px] font-mono text-slate-400">
            <span>{activeSnippet.filename}</span>
            <span>Language: {activeSnippet.language}</span>
          </div>
          <pre className="p-4 text-xs font-mono overflow-x-auto leading-relaxed max-h-[500px]">
            <code>{activeSnippet.code}</code>
          </pre>
        </div>
      ) : (
        /* Side by Side Diff */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Eager side */}
          <div className="rounded-lg overflow-hidden border border-rose-900/60 bg-slate-950 text-slate-200 text-xs font-mono">
            <div className="px-3 py-2 bg-rose-950/40 border-b border-rose-900/50 text-rose-300 font-bold flex justify-between">
              <span>Unfused PyTorch Eager (3 Dispatches)</span>
              <span>- 3x HBM DRAM Round-Trips</span>
            </div>
            <pre className="p-3 text-[11px] overflow-x-auto leading-relaxed text-rose-200/90">
              <code>{`# 1. Residual Add Kernel
h = x + residual  # Writes to HBM

# 2. LayerNorm Kernel
# Reads h from HBM, writes norm to HBM
norm = F.layer_norm(
    h, (D,), weight=gamma, bias=beta
)

# 3. SiLU Activation Kernel
# Reads norm from HBM, writes out to HBM
out = F.silu(norm)

# Result:
# Total DRAM Traffic: ~7.2 * N bytes
# Latency: ~440 μs
# GPU Memory Bus Stall: 64.2%`}</code>
            </pre>
          </div>

          {/* Fused side */}
          <div className="rounded-lg overflow-hidden border border-emerald-900/60 bg-slate-950 text-slate-200 text-xs font-mono">
            <div className="px-3 py-2 bg-emerald-950/40 border-b border-emerald-900/50 text-emerald-300 font-bold flex justify-between">
              <span>Custom Triton / CUDA Kernel (1 Dispatch)</span>
              <span>+ Zero Intermediate Writes</span>
            </div>
            <pre className="p-3 text-[11px] overflow-x-auto leading-relaxed text-emerald-200/90">
              <code>{`# Single Fused Kernel Launch:
# 1. Load X & RES directly into SRAM / RF
acc = tl.load(x_ptrs) + tl.load(res_ptrs)

# 2. Reduction in Registers via Warp Shuffle
mean = tl.sum(acc, axis=0) / N
var = tl.sum((acc - mean)**2, axis=0) / N
rstd = 1.0 / tl.sqrt(var + eps)
norm = (acc - mean) * rstd * gamma + beta

# 3. Fast SiLU computed in ALU register
silu = norm * tl.sigmoid(norm)

# 4. Single coalesced DRAM store
tl.store(out_ptrs, silu)

# Result:
# Total DRAM Traffic: ~3.1 * N bytes (-57%)
# Speedup: 1.82x
# Active Instruction Issue: 38.0%`}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
