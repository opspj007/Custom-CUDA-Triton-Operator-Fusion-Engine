import React, { useState, useMemo } from 'react';
import {
  Binary,
  Cpu,
  Layers,
  Zap,
  Activity,
  ArrowRight,
  GitCompare,
  CheckCircle2,
  AlertTriangle,
  Info,
  Maximize2,
  Search,
} from 'lucide-react';
import {
  getAssemblyData,
  AssemblyInstruction,
  AssemblyRepresentation,
} from '../core/assembly_simulation';

export const AssemblyViewer: React.FC = () => {
  const [selectedKernel, setSelectedKernel] = useState<'unfused' | 'triton' | 'cuda'>('triton');
  const [selectedIsa, setSelectedIsa] = useState<'ptx' | 'sass'>('sass');
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [compareMode, setCompareMode] = useState<boolean>(false);

  const assemblyData: AssemblyRepresentation = useMemo(() => {
    return getAssemblyData(selectedKernel, selectedIsa);
  }, [selectedKernel, selectedIsa]);

  const unfusedData: AssemblyRepresentation = useMemo(() => {
    return getAssemblyData('unfused', selectedIsa);
  }, [selectedIsa]);

  const filteredInstructions = useMemo(() => {
    if (categoryFilter === 'all') return assemblyData.instructions;
    return assemblyData.instructions.filter((inst) => inst.category === categoryFilter);
  }, [assemblyData, categoryFilter]);

  const activeInstruction = useMemo(() => {
    if (selectedLine === null) {
      return assemblyData.instructions.find((i) => i.category === 'memory' || i.category === 'alu') || assemblyData.instructions[0];
    }
    return (
      assemblyData.instructions.find((i) => i.lineNum === selectedLine) ||
      assemblyData.instructions[0]
    );
  }, [assemblyData, selectedLine]);

  const getCategoryColor = (cat: AssemblyInstruction['category']) => {
    switch (cat) {
      case 'memory':
        return 'text-amber-600 dark:text-amber-400 border-amber-500/30 bg-amber-50 dark:bg-amber-950/40';
      case 'alu':
        return 'text-emerald-600 dark:text-emerald-400 border-emerald-500/30 bg-emerald-50 dark:bg-emerald-950/40';
      case 'shuffle':
        return 'text-blue-600 dark:text-blue-400 border-blue-500/30 bg-blue-50 dark:bg-blue-950/40';
      case 'sync':
        return 'text-purple-600 dark:text-purple-400 border-purple-500/30 bg-purple-50 dark:bg-purple-950/40';
      case 'control':
        return 'text-slate-600 dark:text-slate-400 border-slate-500/30 bg-slate-50 dark:bg-slate-800/40';
      default:
        return 'text-slate-500 border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header Card */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center">
                <Binary className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 tracking-tight">
                  PTX &amp; SASS Assembly Instruction-Level Parallelism (ILP) Visualizer
                </h2>
                <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono">
                  <span>Target: Ada Lovelace SM 8.9 &amp; Ampere SM 8.0</span>
                  <span>·</span>
                  <span>CUDA Compiler nvcc 12.4 &amp; Triton JIT MLIR</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Action Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors flex items-center gap-1.5 ${
                compareMode
                  ? 'bg-emerald-50 dark:bg-emerald-950/50 border-emerald-500 text-emerald-700 dark:text-emerald-300'
                  : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
            >
              <GitCompare className="w-3.5 h-3.5" />
              <span>{compareMode ? 'Exit Diff View' : 'Side-by-Side Diff vs Unfused'}</span>
            </button>
          </div>
        </div>

        {/* Engine and ISA Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-4">
          {/* Kernel Target */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">
              Kernel Implementation Target
            </label>
            <div className="grid grid-cols-3 gap-1 bg-slate-100 dark:bg-slate-800/80 p-1 rounded-lg">
              <button
                onClick={() => setSelectedKernel('unfused')}
                className={`px-2 py-1.5 text-xs font-medium rounded-md transition-colors text-center ${
                  selectedKernel === 'unfused'
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                Unfused
              </button>
              <button
                onClick={() => setSelectedKernel('triton')}
                className={`px-2 py-1.5 text-xs font-medium rounded-md transition-colors text-center ${
                  selectedKernel === 'triton'
                    ? 'bg-emerald-600 text-white shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                Triton Fused
              </button>
              <button
                onClick={() => setSelectedKernel('cuda')}
                className={`px-2 py-1.5 text-xs font-medium rounded-md transition-colors text-center ${
                  selectedKernel === 'cuda'
                    ? 'bg-emerald-600 text-white shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                CUDA Fused
              </button>
            </div>
          </div>

          {/* Assembly ISA Format */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">
              Assembly ISA Layer
            </label>
            <div className="grid grid-cols-2 gap-1 bg-slate-100 dark:bg-slate-800/80 p-1 rounded-lg">
              <button
                onClick={() => setSelectedIsa('sass')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors text-center ${
                  selectedIsa === 'sass'
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                SASS Machine Code
              </button>
              <button
                onClick={() => setSelectedIsa('ptx')}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors text-center ${
                  selectedIsa === 'ptx'
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                PTX Virtual ISA
              </button>
            </div>
          </div>

          {/* Opcode Category Filter */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">
              Instruction Filter
            </label>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full bg-slate-100 dark:bg-slate-800 border-none rounded-lg px-3 py-2 text-xs font-medium text-slate-900 dark:text-slate-100 focus:ring-1 focus:ring-emerald-500"
            >
              <option value="all">All Instructions ({assemblyData.instructions.length})</option>
              <option value="memory">Memory LSU (Loads &amp; Stores)</option>
              <option value="alu">Math ALU / FMA / FP32</option>
              <option value="shuffle">Intra-Warp Shuffle (SHFL)</option>
              <option value="sync">Sync Barriers</option>
              <option value="control">Control &amp; Boundaries</option>
            </select>
          </div>

          {/* Target Architecture Info */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">
              Hardware Microarchitecture
            </label>
            <div className="bg-slate-100 dark:bg-slate-800/80 px-3 py-2 rounded-lg text-xs font-mono flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-300">NVIDIA SM 8.9 (Ada)</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Dual-Warp Sched</span>
            </div>
          </div>
        </div>
      </div>

      {/* KPI ILP Metrics Strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
          <div className="text-[11px] font-mono uppercase text-slate-500 dark:text-slate-400">
            Instructions Per Cycle (IPC)
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1 flex items-baseline gap-1.5">
            <span>{assemblyData.ipc.toFixed(2)}</span>
            <span className="text-xs font-normal text-slate-400">/ 4.0 max</span>
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {selectedKernel === 'unfused' ? (
              <span className="text-amber-500">Throttled by DRAM round-trip latency</span>
            ) : (
              <span className="text-emerald-500 font-medium">
                +{(assemblyData.ipc / unfusedData.ipc).toFixed(1)}x instruction throughput
              </span>
            )}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
          <div className="text-[11px] font-mono uppercase text-slate-500 dark:text-slate-400">
            Dual-Issue Rate
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1">
            {assemblyData.dualIssuePercent}%
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Concurrent FMA + LSU execution slots
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
          <div className="text-[11px] font-mono uppercase text-slate-500 dark:text-slate-400">
            RAW Hazard Avg Stall
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
            {assemblyData.rawStallCyclesAvg}{' '}
            <span className="text-xs font-normal text-slate-400">cycles</span>
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {selectedKernel === 'unfused'
              ? '420 cyc DRAM scoreboard wait'
              : 'Kept in RF (12 cyc scoreboard)'}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs">
          <div className="text-[11px] font-mono uppercase text-slate-500 dark:text-slate-400">
            Global Memory Transfers
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-slate-100 mt-1">
            {assemblyData.memoryOpsCount}{' '}
            <span className="text-xs font-normal text-slate-400">ops</span>
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {selectedKernel === 'unfused'
              ? '6 scalar DRAM round-trips'
              : 'Vectorized 128-bit loads/stores'}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 shadow-xs col-span-2 md:col-span-1">
          <div className="text-[11px] font-mono uppercase text-slate-500 dark:text-slate-400">
            ILP Acceleration
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-1">
            {assemblyData.ilpSpeedupScore.toFixed(2)}x
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Pipeline efficiency vs eager
          </div>
        </div>
      </div>

      {/* Main Assembly & Pipeline Analysis Workstation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Assembly Code Listing (7 cols or 6 cols if compare mode) */}
        <div className={compareMode ? 'lg:col-span-6 space-y-4' : 'lg:col-span-7 space-y-4'}>
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-xs">
            <div className="bg-slate-100 dark:bg-slate-800/70 px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-slate-800 dark:text-slate-200">
                  {assemblyData.kernelName}
                </span>
                <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
                  {assemblyData.isa}
                </span>
              </div>
              <span className="text-[11px] text-slate-500 font-mono">
                {filteredInstructions.length} instructions
              </span>
            </div>

            {/* Instruction List Table */}
            <div className="overflow-x-auto max-h-[580px] divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-xs">
              {filteredInstructions.map((inst) => {
                const isSelected = activeInstruction.lineNum === inst.lineNum;
                return (
                  <div
                    key={inst.lineNum}
                    onClick={() => setSelectedLine(inst.lineNum)}
                    className={`px-4 py-2.5 flex items-center gap-3 cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-emerald-50 dark:bg-emerald-950/40 border-l-4 border-emerald-500'
                        : 'hover:bg-slate-50 dark:hover:bg-slate-800/40 border-l-4 border-transparent'
                    }`}
                  >
                    <span className="w-8 shrink-0 text-slate-400 text-right select-none text-[11px]">
                      {inst.lineNum}
                    </span>

                    {/* Instruction String with syntax highlight approximation */}
                    <div className="flex-1 font-mono font-medium truncate text-slate-800 dark:text-slate-200">
                      {inst.instruction.startsWith('//') ? (
                        <span className="text-slate-400 italic">{inst.instruction}</span>
                      ) : (
                        <span>
                          <span className="text-emerald-700 dark:text-emerald-400 font-bold">
                            {inst.instruction.split(' ')[0]}
                          </span>{' '}
                          <span className="text-slate-700 dark:text-slate-300">
                            {inst.instruction.split(' ').slice(1).join(' ')}
                          </span>
                        </span>
                      )}
                    </div>

                    {/* Unit & Latency Badges */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      {inst.dualIssueEligible && (
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300 font-semibold" title="Can be issued concurrently with another sub-core instruction in the same cycle">
                          Dual-Issue
                        </span>
                      )}

                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded border ${getCategoryColor(
                          inst.category
                        )}`}
                      >
                        {inst.unit.split(' ')[0]}
                      </span>

                      <span className="text-[10px] font-mono text-slate-400 w-16 text-right">
                        {inst.cyclesLatency > 0 ? `~${inst.cyclesLatency} cyc` : '0 cyc'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Diff View Comparison (when compareMode is active) */}
        {compareMode && (
          <div className="lg:col-span-6 space-y-4">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-xs">
              <div className="bg-amber-50 dark:bg-amber-950/40 px-4 py-3 border-b border-amber-200 dark:border-amber-900/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold text-amber-900 dark:text-amber-200">
                    Unfused Baseline (Reference Comparison)
                  </span>
                  <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-amber-200 dark:bg-amber-900 text-amber-800 dark:text-amber-300">
                    {unfusedData.isa}
                  </span>
                </div>
                <span className="text-[11px] text-amber-700 dark:text-amber-400 font-mono">
                  {unfusedData.instructions.length} instructions (IPC: {unfusedData.ipc})
                </span>
              </div>

              <div className="overflow-x-auto max-h-[580px] divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-xs">
                {unfusedData.instructions.map((inst) => (
                  <div
                    key={inst.lineNum}
                    className="px-4 py-2.5 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-800/40"
                  >
                    <span className="w-8 shrink-0 text-slate-400 text-right select-none text-[11px]">
                      {inst.lineNum}
                    </span>
                    <div className="flex-1 font-mono truncate text-slate-700 dark:text-slate-300">
                      {inst.instruction.startsWith('//') ? (
                        <span className="text-slate-400 italic">{inst.instruction}</span>
                      ) : (
                        <span>
                          <span className="text-amber-600 dark:text-amber-400 font-semibold">
                            {inst.instruction.split(' ')[0]}
                          </span>{' '}
                          {inst.instruction.split(' ').slice(1).join(' ')}
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400 w-16 text-right">
                      {inst.cyclesLatency > 0 ? `~${inst.cyclesLatency} cyc` : '0 cyc'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Right Column: Instruction Inspector & Hardware ILP Pipeline (5 cols or full width in bottom) */}
        <div className={compareMode ? 'lg:col-span-12 space-y-4' : 'lg:col-span-5 space-y-4'}>
          {/* Active Instruction Deep-Dive Card */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-emerald-600" />
                <h3 className="font-semibold text-xs text-slate-900 dark:text-slate-100 uppercase tracking-wider font-mono">
                  Instruction Inspector
                </h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                Line {activeInstruction.lineNum}
              </span>
            </div>

            <div className="bg-slate-900 text-emerald-400 p-3 rounded-lg font-mono text-xs overflow-x-auto">
              <code>{activeInstruction.instruction}</code>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-50 dark:bg-slate-800/50 p-2.5 rounded-lg">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">
                  Execution Sub-Core
                </span>
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  {activeInstruction.unit}
                </span>
              </div>
              <div className="bg-slate-50 dark:bg-slate-800/50 p-2.5 rounded-lg">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">
                  Pipeline Latency
                </span>
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  {activeInstruction.cyclesLatency} cycles
                </span>
              </div>
              <div className="bg-slate-50 dark:bg-slate-800/50 p-2.5 rounded-lg">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">
                  Dual-Issue Slot
                </span>
                <span
                  className={`font-semibold ${
                    activeInstruction.dualIssueEligible
                      ? 'text-emerald-600 dark:text-emerald-400'
                      : 'text-slate-500'
                  }`}
                >
                  {activeInstruction.dualIssueEligible ? 'Eligible (Slot 0 / Slot 1)' : 'Single-Issue'}
                </span>
              </div>
              <div className="bg-slate-50 dark:bg-slate-800/50 p-2.5 rounded-lg">
                <span className="text-[10px] font-mono text-slate-400 block uppercase">
                  Dataflow Class
                </span>
                <span className="font-medium uppercase text-slate-800 dark:text-slate-200">
                  {activeInstruction.category}
                </span>
              </div>
            </div>

            <div className="text-xs text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800/30 p-3 rounded-lg border border-slate-100 dark:border-slate-800">
              <div className="font-semibold text-[11px] font-mono uppercase text-slate-400 mb-1">
                Hardware Architectural Impact
              </div>
              <p className="leading-relaxed">{activeInstruction.explanation}</p>
            </div>
          </div>

          {/* Warp Scheduler Dual-Dispatch ILP Slot Visualization */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-600" />
                <h3 className="font-semibold text-xs text-slate-900 dark:text-slate-100 uppercase tracking-wider font-mono">
                  Warp Issue Pipeline Simulation
                </h3>
              </div>
              <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
                Ada SM 8.9 Warp Sched
              </span>
            </div>

            <p className="text-xs text-slate-500 leading-relaxed">
              In Ada &amp; Ampere SMs, each sub-partition warp scheduler issues up to 2 independent instructions per warp per cycle (Dual-Issue ILP), provided there are no RAW stalls.
            </p>

            {/* Pipeline Slots Diagram */}
            <div className="space-y-2 font-mono text-xs">
              <div className="bg-slate-50 dark:bg-slate-800/50 p-2.5 rounded-lg border border-slate-200 dark:border-slate-700">
                <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1.5">
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">Cycle N: Dual Dispatch</span>
                  <span>Slot 0: ALU (Math) · Slot 1: LSU (Mem)</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-emerald-100/70 dark:bg-emerald-950/60 text-emerald-900 dark:text-emerald-300 px-2 py-1.5 rounded border border-emerald-300 dark:border-emerald-800 text-[11px] truncate">
                    FMA / FMUL R20, R4, R12
                  </div>
                  <div className="bg-amber-100/70 dark:bg-amber-950/60 text-amber-900 dark:text-amber-300 px-2 py-1.5 rounded border border-amber-300 dark:border-amber-800 text-[11px] truncate">
                    LDG.E.128.SYS [R30]
                  </div>
                </div>
              </div>

              {selectedKernel === 'unfused' ? (
                <div className="bg-rose-50 dark:bg-rose-950/40 p-2.5 rounded-lg border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-300 text-xs">
                  <div className="flex items-center gap-1.5 font-bold mb-1">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Unfused Pipeline Stall (Cycles N+1 ... N+450)</span>
                  </div>
                  <p className="text-[11px] text-rose-700 dark:text-rose-400">
                    Warp scheduler enters Long Scoreboard stall. 0 instructions dispatched for 420+ cycles waiting for scalar DRAM round-trips.
                  </p>
                </div>
              ) : (
                <div className="bg-emerald-50 dark:bg-emerald-950/40 p-2.5 rounded-lg border border-emerald-200 dark:border-emerald-900 text-emerald-800 dark:text-emerald-300 text-xs">
                  <div className="flex items-center gap-1.5 font-bold mb-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Fused Continuous Execution (Cycles N+1 ... N+4)</span>
                  </div>
                  <p className="text-[11px] text-emerald-700 dark:text-emerald-400">
                    Intermediates remain in Register File (RF). RAW latency is only 4 cycles for FMA and 1 cycle for intra-warp shuffle, eliminating pipeline bubbles.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 4 Pillars of Fusion ILP Acceleration Guide */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-600" />
          How Operator Fusion Unlocks Instruction-Level Parallelism (ILP) in NVIDIA SASS
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 flex items-center justify-center font-mono text-[11px]">
                1
              </span>
              Register-File Pinning
            </div>
            <p className="text-slate-500 leading-relaxed">
              Unfused PyTorch emits 6 isolated DRAM transactions (`LDG`/`STG`). Fusion holds `add_out` in hardware registers (`R20..R23`), slashing Long Scoreboard wait cycles from 420 to 0.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 flex items-center justify-center font-mono text-[11px]">
                2
              </span>
              128-Bit Memory Vectorization
            </div>
            <p className="text-slate-500 leading-relaxed">
              nvcc and Triton lower loads to `LDG.E.128.SYS`, fetching four FP32 elements in one transaction. This cuts LSU instruction pressure by 4x, leaving warp schedulers free to dual-issue math.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 flex items-center justify-center font-mono text-[11px]">
                3
              </span>
              Intra-Warp Reduction Shuffles
            </div>
            <p className="text-slate-500 leading-relaxed">
              LayerNorm mean and variance reductions are executed using `SHFL.IDX` / `__shfl_down_sync`. Threads exchange data in 1 clock cycle without touching shared memory or global memory.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 flex items-center justify-center font-mono text-[11px]">
                4
              </span>
              Dual-Issue Instruction Packing
            </div>
            <p className="text-slate-500 leading-relaxed">
              With memory stalls eliminated, the Ada Lovelace dual-dispatch warp scheduler issues simultaneous FP32 FMA arithmetic and LSU memory operations, reaching 2.85 IPC (73.8% dual-issue).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
