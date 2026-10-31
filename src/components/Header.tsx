import React from 'react';
import { Cpu, Play, Download, Terminal, Layers } from 'lucide-react';
import { GpuSpec } from '../core/gpu_simulation';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  activeGpu: GpuSpec;
  isSimulating: boolean;
  onRunProfile: () => void;
  onExportReport: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  activeGpu,
  isSimulating,
  onRunProfile,
  onExportReport,
}) => {
  const tabs = [
    { id: 'stages', label: 'Project Stages (1-6)' },
    { id: 'studio', label: 'Kernel Studio' },
    { id: 'memory', label: 'Memory Pipeline' },
    { id: 'nsight', label: 'Nsight Profiler' },
    { id: 'assembly', label: 'PTX / SASS Assembly' },
    { id: 'roofline', label: 'Roofline Model' },
    { id: 'verification', label: 'Numeric Verification' },
    { id: 'code', label: 'Source Code' },
    { id: 'sweeps', label: 'Benchmark Sweeps' },
  ];

  return (
    <header className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 sticky top-0 z-40 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          {/* Zone 1: Brand Wordmark */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-9 h-9 rounded-lg bg-emerald-600 text-white flex items-center justify-center font-bold shadow-sm">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-slate-900 dark:text-slate-100 tracking-tight whitespace-nowrap">
                  CUDA · TRITON FUSION ENGINE
                </span>
                <span className="hidden sm:inline-block text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                  {activeGpu.computeCapability}
                </span>
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Transformer Block Operator Fusion · LayerNorm + Residual Add + SiLU
              </div>
            </div>
          </div>

          {/* Zone 2: Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1 bg-slate-100 dark:bg-slate-800/70 p-1 rounded-lg">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap shrink-0 ${
                    isActive
                      ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: Primary Actions */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={onExportReport}
              title="Export Benchmark JSON"
              aria-label="Export Benchmark JSON"
              className="px-3 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden md:inline">Export Metrics</span>
            </button>
            <button
              onClick={onRunProfile}
              disabled={isSimulating}
              className="px-4 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 rounded-lg transition-colors flex items-center gap-1.5 shadow-xs whitespace-nowrap shrink-0 disabled:opacity-50"
            >
              <Play className={`w-3.5 h-3.5 ${isSimulating ? 'animate-spin' : ''}`} />
              <span>{isSimulating ? 'Profiling Kernel...' : 'Execute Profile Run'}</span>
            </button>
          </div>
        </div>

        {/* Mobile Tab Selector */}
        <div className="lg:hidden flex items-center gap-1 overflow-x-auto py-2 border-t border-slate-200 dark:border-slate-800">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap shrink-0 ${
                activeTab === tab.id
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
