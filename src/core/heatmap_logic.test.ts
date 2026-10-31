import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  generateShapeHeatmapGrid,
  calculateWaveEfficiency,
  getHeatmapCellColor,
  HEATMAP_BATCH_SIZES,
  HEATMAP_SEQ_LENGTHS,
} from './heatmap_logic.ts';

describe('Heatmap Logic & Sweet Spot Analysis', () => {
  it('generates a complete 7x7 matrix matching batch sizes and seq lengths', () => {
    const matrix = generateShapeHeatmapGrid({
      gpuId: 'rtx4090',
      hiddenDim: 2048,
      precision: 'FP16',
    });

    assert.strictEqual(matrix.batchSizes.length, HEATMAP_BATCH_SIZES.length);
    assert.strictEqual(matrix.seqLengths.length, HEATMAP_SEQ_LENGTHS.length);
    assert.strictEqual(matrix.cells.length, 7);
    assert.strictEqual(matrix.cells[0].length, 7);
    assert.ok(matrix.sweetSpotCell);
    assert.ok(matrix.sweetSpotCell.isSweetSpot);
  });

  it('identifies sweet spot with high speedup and realistic token footprint', () => {
    const matrix = generateShapeHeatmapGrid({
      gpuId: 'rtx4090',
      hiddenDim: 2048,
      precision: 'FP16',
    });

    const sweet = matrix.sweetSpotCell;
    assert.ok(sweet.tritonSpeedup >= 1.5, `Speedup was ${sweet.tritonSpeedup}, expected >= 1.5`);
    assert.ok(sweet.waveEfficiencyPercent >= 70, `Wave efficiency was ${sweet.waveEfficiencyPercent}, expected >= 70%`);
    assert.ok(sweet.tokens >= 1024, `Tokens was ${sweet.tokens}, expected >= 1024`);
    assert.ok(sweet.reason && sweet.reason.length > 10);
  });

  it('correctly models wave quantization efficiency', () => {
    // 128 SMs on RTX 4090, 512 block size
    // 128 blocks * 512 = 65,536 tokens -> exactly 1 wave across 128 SMs -> high efficiency
    const exactWaveEff = calculateWaveEfficiency(65536, 512, 128);
    assert.ok(exactWaveEff >= 95, `Expected >= 95%, got ${exactWaveEff}%`);

    // Under-saturated small tokens (e.g. 512 tokens = 1 block on 128 SMs) -> low efficiency
    const tinyWaveEff = calculateWaveEfficiency(512, 512, 128);
    assert.ok(tinyWaveEff < 50, `Expected < 50%, got ${tinyWaveEff}%`);
  });

  it('generates valid styling and display values across all metric modes', () => {
    const matrix = generateShapeHeatmapGrid({
      gpuId: 'a100_sxm',
      hiddenDim: 4096,
      precision: 'BF16',
    });

    const cell = matrix.cells[3][3];
    const modes: Array<'tritonSpeedup' | 'cudaSpeedup' | 'tritonLatency' | 'cudaLatency' | 'eagerLatency' | 'dramSaved'> = [
      'tritonSpeedup',
      'cudaSpeedup',
      'tritonLatency',
      'cudaLatency',
      'eagerLatency',
      'dramSaved',
    ];

    for (const mode of modes) {
      const color = getHeatmapCellColor(cell, mode, matrix);
      assert.ok(color.backgroundColor.startsWith('rgba('));
      assert.ok(color.textColor.length > 0);
      assert.ok(color.displayValue.length > 0);
    }
  });
});
