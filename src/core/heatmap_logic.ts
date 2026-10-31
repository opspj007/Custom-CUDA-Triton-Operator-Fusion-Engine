import {
  calculateTheoreticalProfile,
  GPU_CATALOG,
  GpuSpec,
  PrecisionType,
  KernelExecutionConfig,
} from './gpu_simulation';

export const HEATMAP_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64] as const;
export const HEATMAP_SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096] as const;

export type HeatmapMetricMode = 'tritonSpeedup' | 'cudaSpeedup' | 'tritonLatency' | 'cudaLatency' | 'eagerLatency' | 'dramSaved';

export interface HeatmapCell {
  batchSize: number;
  seqLength: number;
  tokens: number;
  eagerLatencyUs: number;
  unfusedCudaLatencyUs: number;
  inductorLatencyUs: number;
  tritonLatencyUs: number;
  cudaLatencyUs: number;
  tritonSpeedup: number;
  cudaSpeedup: number;
  inductorSpeedup: number;
  eagerDramMB: number;
  fusedDramMB: number;
  dramSavedMB: number;
  waveEfficiencyPercent: number; // Wave quantization efficiency on target SMs
  isSweetSpot: boolean;
  sweetSpotScore: number;
  reason?: string;
}

export interface HeatmapMatrix {
  gpu: GpuSpec;
  hiddenDim: number;
  precision: PrecisionType;
  batchSizes: number[];
  seqLengths: number[];
  cells: HeatmapCell[][]; // [batchIdx][seqIdx]
  sweetSpotCell: HeatmapCell;
  minSpeedup: number;
  maxSpeedup: number;
  minLatencyUs: number;
  maxLatencyUs: number;
  maxDramSavedMB: number;
}

/**
 * Calculates wave quantization efficiency:
 * Given total thread blocks = ceil(totalTokens / blockSize),
 * calculates how evenly blocks distribute across GPU SMs.
 */
export function calculateWaveEfficiency(
  tokens: number,
  blockSize: number,
  smCount: number
): number {
  const blocks = Math.ceil(tokens / blockSize);
  const waves = blocks / smCount;
  const fullWaves = Math.floor(waves);
  const remainder = waves - fullWaves;
  
  if (fullWaves === 0) {
    // Under-saturating the GPU SMs in first wave
    return Math.max(30, Math.round(waves * 100));
  }
  // If remainder is very small, nearly 100% full wave
  if (remainder < 0.05) return 98;
  // Fractional tail wave penalty
  const tailPenalty = (1 - remainder) * 0.25;
  return Math.round((1 - tailPenalty) * 100);
}

/**
 * Generates the full 2D Latency & Fusion Performance Heatmap
 * across (Batch Size x Sequence Length).
 */
export function generateShapeHeatmapGrid(params: {
  gpuId: string;
  hiddenDim: number;
  precision: PrecisionType;
  tritonBlockSize?: number;
  numWarps?: number;
}): HeatmapMatrix {
  const { gpuId, hiddenDim, precision } = params;
  const gpu = GPU_CATALOG[gpuId] || GPU_CATALOG['rtx4090'];
  const blockSize = params.tritonBlockSize || 512;
  const numWarps = params.numWarps || 8;

  const batchSizes = [...HEATMAP_BATCH_SIZES];
  const seqLengths = [...HEATMAP_SEQ_LENGTHS];

  let minSpeedup = Infinity;
  let maxSpeedup = -Infinity;
  let minLatencyUs = Infinity;
  let maxLatencyUs = -Infinity;
  let maxDramSavedMB = -Infinity;

  const rawCells: HeatmapCell[][] = [];
  let bestScore = -1;
  let sweetSpotCoord = { bIdx: 0, sIdx: 0 };

  for (let bIdx = 0; bIdx < batchSizes.length; bIdx++) {
    const row: HeatmapCell[] = [];
    const b = batchSizes[bIdx];

    for (let sIdx = 0; sIdx < seqLengths.length; sIdx++) {
      const s = seqLengths[sIdx];
      const tokens = b * s;

      const cellConfig: KernelExecutionConfig = {
        gpuId,
        batchSize: b,
        seqLength: s,
        hiddenDim,
        precision,
        tritonBlockSize: blockSize,
        numWarps,
        numStages: 3,
      };

      const prof = calculateTheoreticalProfile(cellConfig);

      const eagerLat = prof.eagerResult.latencyUs;
      const unfusedCudaLat = prof.unfusedCudaResult.latencyUs;
      const inductorLat = prof.inductorResult.latencyUs;
      const tritonLat = prof.tritonResult.latencyUs;
      const cudaLat = prof.cudaResult.latencyUs;

      const tritonSpd = prof.tritonResult.speedupVsEager;
      const cudaSpd = prof.cudaResult.speedupVsEager;
      const inductorSpd = prof.inductorResult.speedupVsEager;

      const eagerMB = prof.eagerResult.dramTrafficMB;
      const fusedMB = prof.tritonResult.dramTrafficMB;
      const dramSaved = Math.max(0, Number((eagerMB - fusedMB).toFixed(2)));

      const waveEff = calculateWaveEfficiency(tokens, blockSize, gpu.smCount);

      if (tritonSpd < minSpeedup) minSpeedup = tritonSpd;
      if (tritonSpd > maxSpeedup) maxSpeedup = tritonSpd;
      if (tritonLat < minLatencyUs) minLatencyUs = tritonLat;
      if (tritonLat > maxLatencyUs) maxLatencyUs = tritonLat;
      if (dramSaved > maxDramSavedMB) maxDramSavedMB = dramSaved;

      // Sweet Spot Score: Balances high speedup, high wave efficiency, and realistic batch size for inference/training
      // Penalizes tiny batches (<1024 tokens) where launch overhead masks benefits, and rewards shapes with full wave saturation
      let sweetScore = tritonSpd * (waveEff / 100);
      if (tokens >= 4096 && tokens <= 32768) {
        sweetScore += 0.4; // Golden Transformer serving zone
      }
      if (b >= 4 && b <= 32 && s >= 256 && s <= 2048) {
        sweetScore += 0.25;
      }

      if (sweetScore > bestScore) {
        bestScore = sweetScore;
        sweetSpotCoord = { bIdx, sIdx };
      }

      row.push({
        batchSize: b,
        seqLength: s,
        tokens,
        eagerLatencyUs: eagerLat,
        unfusedCudaLatencyUs: unfusedCudaLat,
        inductorLatencyUs: inductorLat,
        tritonLatencyUs: tritonLat,
        cudaLatencyUs: cudaLat,
        tritonSpeedup: tritonSpd,
        cudaSpeedup: cudaSpd,
        inductorSpeedup: inductorSpd,
        eagerDramMB: eagerMB,
        fusedDramMB: fusedMB,
        dramSavedMB: dramSaved,
        waveEfficiencyPercent: waveEff,
        isSweetSpot: false,
        sweetSpotScore: sweetScore,
      });
    }
    rawCells.push(row);
  }

  // Mark the sweet spot cell
  const sweetCell = rawCells[sweetSpotCoord.bIdx][sweetSpotCoord.sIdx];
  sweetCell.isSweetSpot = true;
  sweetCell.reason = `Optimal SM wave saturation (${sweetCell.waveEfficiencyPercent}%) on ${gpu.name} (${gpu.smCount} SMs), eliminating ${sweetCell.dramSavedMB.toLocaleString()} MB DRAM traffic with ${sweetCell.tritonSpeedup.toFixed(2)}× speedup.`;

  return {
    gpu,
    hiddenDim,
    precision,
    batchSizes,
    seqLengths,
    cells: rawCells,
    sweetSpotCell: sweetCell,
    minSpeedup,
    maxSpeedup,
    minLatencyUs,
    maxLatencyUs,
    maxDramSavedMB,
  };
}

/**
 * Returns color style classes and inline hex colors for a heatmap cell based on the selected metric mode.
 */
export function getHeatmapCellColor(
  cell: HeatmapCell,
  mode: HeatmapMetricMode,
  matrix: HeatmapMatrix
): {
  backgroundColor: string;
  textColor: string;
  borderColor?: string;
  displayValue: string;
} {
  switch (mode) {
    case 'tritonSpeedup': {
      const spd = cell.tritonSpeedup;
      const min = matrix.minSpeedup;
      const max = matrix.maxSpeedup;
      const range = max - min || 1;
      const ratio = Math.max(0, Math.min(1, (spd - min) / range));

      // Emerald gradient from subtle dark green to glowing emerald
      let bg = `rgba(16, 185, 129, ${0.15 + ratio * 0.75})`;
      let text = ratio > 0.4 ? '#ffffff' : '#047857';

      return {
        backgroundColor: bg,
        textColor: text,
        displayValue: `${spd.toFixed(2)}×`,
      };
    }

    case 'cudaSpeedup': {
      const spd = cell.cudaSpeedup;
      const min = matrix.minSpeedup;
      const max = matrix.maxSpeedup * 1.05;
      const range = max - min || 1;
      const ratio = Math.max(0, Math.min(1, (spd - min) / range));

      let bg = `rgba(5, 150, 105, ${0.15 + ratio * 0.8})`;
      let text = ratio > 0.4 ? '#ffffff' : '#047857';

      return {
        backgroundColor: bg,
        textColor: text,
        displayValue: `${spd.toFixed(2)}×`,
      };
    }

    case 'tritonLatency': {
      const lat = cell.tritonLatencyUs;
      const min = matrix.minLatencyUs;
      const max = matrix.maxLatencyUs;
      const range = max - min || 1;
      const ratio = Math.max(0, Math.min(1, (lat - min) / range));

      // From deep blue/cyan (lowest latency) to warmer purple/slate (higher latency)
      let bg = `rgba(59, 130, 246, ${0.12 + ratio * 0.78})`;
      let text = ratio > 0.45 ? '#ffffff' : '#1e40af';

      return {
        backgroundColor: bg,
        textColor: text,
        displayValue: lat < 1000 ? `${lat}μs` : `${(lat / 1000).toFixed(1)}ms`,
      };
    }

    case 'cudaLatency': {
      const lat = cell.cudaLatencyUs;
      const min = matrix.minLatencyUs * 0.95;
      const max = matrix.maxLatencyUs * 0.95;
      const range = max - min || 1;
      const ratio = Math.max(0, Math.min(1, (lat - min) / range));

      let bg = `rgba(14, 165, 233, ${0.12 + ratio * 0.78})`;
      let text = ratio > 0.45 ? '#ffffff' : '#0369a1';

      return {
        backgroundColor: bg,
        textColor: text,
        displayValue: lat < 1000 ? `${lat}μs` : `${(lat / 1000).toFixed(1)}ms`,
      };
    }

    case 'eagerLatency': {
      const lat = cell.eagerLatencyUs;
      const min = matrix.minLatencyUs * 1.5;
      const max = matrix.maxLatencyUs * 2.8;
      const range = max - min || 1;
      const ratio = Math.max(0, Math.min(1, (lat - min) / range));

      let bg = `rgba(239, 68, 68, ${0.12 + ratio * 0.75})`;
      let text = ratio > 0.4 ? '#ffffff' : '#b91c1c';

      return {
        backgroundColor: bg,
        textColor: text,
        displayValue: lat < 1000 ? `${lat}μs` : `${(lat / 1000).toFixed(1)}ms`,
      };
    }

    case 'dramSaved': {
      const saved = cell.dramSavedMB;
      const max = matrix.maxDramSavedMB || 1;
      const ratio = Math.max(0, Math.min(1, saved / max));

      let bg = `rgba(168, 85, 247, ${0.12 + ratio * 0.8})`;
      let text = ratio > 0.4 ? '#ffffff' : '#7e22ce';

      return {
        backgroundColor: bg,
        textColor: text,
        displayValue: saved < 1 ? `${Math.round(saved * 1024)}KB` : `${Math.round(saved)}MB`,
      };
    }
  }
}
