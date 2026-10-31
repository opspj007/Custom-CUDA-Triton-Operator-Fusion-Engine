import { describe, it } from 'node:test';
import assert from 'node:assert';
import { getAssemblyData, ASSEMBLY_DATABASE } from './assembly_simulation.ts';

describe('Assembly Simulation & ILP Analyzer', () => {
  it('returns valid PTX and SASS data for unfused, triton, and cuda kernels', () => {
    const kernels: Array<'unfused' | 'triton' | 'cuda'> = ['unfused', 'triton', 'cuda'];
    const formats: Array<'ptx' | 'sass'> = ['ptx', 'sass'];

    for (const k of kernels) {
      for (const fmt of formats) {
        const data = getAssemblyData(k, fmt);
        assert.ok(data, `Data should exist for ${k} ${fmt}`);
        assert.ok(data.instructions.length > 0, `Instructions should be non-empty for ${k} ${fmt}`);
        assert.ok(data.ipc > 0, `IPC should be positive for ${k} ${fmt}`);
        assert.ok(typeof data.dualIssuePercent === 'number');
        assert.ok(data.totalInstructions > 0);
      }
    }
  });

  it('demonstrates superior ILP and dual-issue rate in fused kernels vs unfused', () => {
    const unfusedSass = getAssemblyData('unfused', 'sass');
    const tritonSass = getAssemblyData('triton', 'sass');
    const cudaSass = getAssemblyData('cuda', 'sass');

    // Fused kernels should achieve significantly higher IPC due to eliminated DRAM roundtrips and vectorization
    assert.ok(tritonSass.ipc > unfusedSass.ipc * 1.5, 'Triton IPC should be >1.5x unfused IPC');
    assert.ok(cudaSass.ipc > unfusedSass.ipc * 1.5, 'CUDA IPC should be >1.5x unfused IPC');

    // Dual-issue percentage should be much higher for fused kernels
    assert.ok(tritonSass.dualIssuePercent > unfusedSass.dualIssuePercent * 2, 'Dual issue % should be higher in fused');
    assert.ok(cudaSass.dualIssuePercent > unfusedSass.dualIssuePercent * 2, 'Dual issue % should be higher in fused');

    // RAW stall cycles average should be drastically lower for fused kernels (avoiding DRAM reloads)
    assert.ok(tritonSass.rawStallCyclesAvg < unfusedSass.rawStallCyclesAvg / 10, 'Triton RAW stalls should be <10% of unfused');
    assert.ok(cudaSass.rawStallCyclesAvg < unfusedSass.rawStallCyclesAvg / 10, 'CUDA RAW stalls should be <10% of unfused');
  });

  it('contains valid instruction metadata categories and execution units', () => {
    const cudaPtx = getAssemblyData('cuda', 'ptx');
    const validCategories = new Set(['memory', 'alu', 'shuffle', 'sync', 'control']);
    const validUnits = new Set(['LSU (Load/Store)', 'FMA (Math)', 'SFU (Special Func)', 'ALU (Logic)', 'Warp Unit']);

    for (const inst of cudaPtx.instructions) {
      assert.ok(validCategories.has(inst.category), `Invalid category: ${inst.category}`);
      assert.ok(validUnits.has(inst.unit), `Invalid unit: ${inst.unit}`);
      assert.ok(typeof inst.cyclesLatency === 'number');
      assert.ok(inst.explanation.length > 5);
    }
  });

  it('handles fallback gracefully for unknown keys', () => {
    // @ts-expect-error Testing fallback runtime resilience
    const fallback = getAssemblyData('invalid_key', 'ptx');
    assert.ok(fallback.instructions.length > 0);
  });
});
