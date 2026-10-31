import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateTheoreticalProfile,
  verifyNumericEquivalence,
  generateHonestReadme,
  GPU_CATALOG,
  KernelExecutionConfig,
} from './gpu_simulation';

test('GPU catalog contains Ada and Ampere architectures with valid bandwidth', () => {
  assert.ok(GPU_CATALOG['rtx4090'], 'RTX 4090 exists');
  assert.equal(GPU_CATALOG['rtx4090'].architecture, 'Ada Lovelace');
  assert.equal(GPU_CATALOG['rtx4090'].memoryBandwidthGBs, 1008);

  assert.ok(GPU_CATALOG['a100_80gb'], 'A100 exists');
  assert.equal(GPU_CATALOG['a100_80gb'].architecture, 'Ampere');
  assert.equal(GPU_CATALOG['a100_80gb'].memoryBandwidthGBs, 2039);
});

test('calculateTheoreticalProfile calculates ~1.8x speedup and significant DRAM reduction', () => {
  const config: KernelExecutionConfig = {
    gpuId: 'rtx4090',
    batchSize: 16,
    seqLength: 512,
    hiddenDim: 2048,
    precision: 'FP16',
    tritonBlockSize: 512,
    numWarps: 8,
    numStages: 3,
  };

  const profile = calculateTheoreticalProfile(config);

  // Check element count: 16 * 512 * 2048 = 16,777,216 elements
  assert.equal(profile.totalElements, 16 * 512 * 2048);

  // Unfused DRAM bytes should be ~8.5x elements * 2 bytes, Fused should be ~3.0x elements * 2 bytes
  assert.ok(profile.unfusedDramBytes > profile.fusedDramBytes, 'Unfused DRAM is greater than Fused DRAM');
  
  // Bandwidth reduction percent should be around 60-65%
  assert.ok(profile.tritonResult.bandwidthReductionPercent > 50, 'Triton reduces bandwidth by >50%');

  // Triton speedup vs Eager should be around 1.7x - 2.1x (target matches ~1.82x)
  assert.ok(profile.tritonResult.speedupVsEager >= 1.5, `Speedup was ${profile.tritonResult.speedupVsEager}, expected >= 1.5`);
  assert.ok(profile.tritonResult.speedupVsEager <= 2.8, `Speedup was ${profile.tritonResult.speedupVsEager}, expected <= 2.8`);

  // Kernel launches: Eager has 3 launches, Fused has 1 launch
  assert.equal(profile.eagerResult.kernelLaunchCount, 3);
  assert.equal(profile.tritonResult.kernelLaunchCount, 1);
  assert.equal(profile.cudaResult.kernelLaunchCount, 1);

  // Memory stall: Eager should have high Long Scoreboard stall (>50%), Fused should be much lower (<30%)
  assert.ok(profile.eagerResult.stallBreakdown.longScoreboardMemStall > 50);
  assert.ok(profile.tritonResult.stallBreakdown.longScoreboardMemStall < 30);
});

test('verifyNumericEquivalence verifies identical mathematical output', () => {
  const check = verifyNumericEquivalence(16);
  assert.ok(check.allMatch, 'Eager vs Fused output matches within epsilon');
  assert.ok(check.cosineSimilarity > 0.999999, 'Cosine similarity is > 0.999999');
  assert.ok(check.maxAbsDiff < 1e-4, 'Max absolute diff is negligible');
});

test('calculateTheoreticalProfile provides statistical timing metrics and unfused CUDA baseline', () => {
  const config: KernelExecutionConfig = {
    gpuId: 'rtx4090',
    batchSize: 16,
    seqLength: 512,
    hiddenDim: 2048,
    precision: 'FP16',
    tritonBlockSize: 512,
    numWarps: 8,
    numStages: 3,
  };

  const profile = calculateTheoreticalProfile(config);

  // Check statistical fields
  assert.ok(profile.tritonResult.medianUs > 0);
  assert.ok(profile.tritonResult.p95Us >= profile.tritonResult.latencyUs);
  assert.ok(profile.tritonResult.minUs <= profile.tritonResult.latencyUs);
  assert.ok(profile.tritonResult.maxUs >= profile.tritonResult.p95Us);

  // Check unfused CUDA baseline
  assert.equal(profile.unfusedCudaResult.engine, 'Unfused CUDA Baseline');
  assert.equal(profile.unfusedCudaResult.kernelLaunchCount, 3);
  assert.ok(profile.unfusedCudaResult.speedupVsEager >= 1.05);

  // Check README generator
  const readme = generateHonestReadme(
    config,
    profile.eagerResult,
    profile.unfusedCudaResult,
    profile.inductorResult,
    profile.tritonResult,
    profile.cudaResult
  );

  assert.ok(readme.includes('Benchmark Result: Custom fused Triton kernel achieved'));
  assert.ok(readme.includes('Stage 1: CUDA C++ Baseline (Unfused)'));
  assert.ok(readme.includes('Stage 2: Triton Fused Kernel'));
  assert.ok(readme.includes('Stage 3: CUDA Fused Kernel'));
  assert.ok(readme.includes('Stage 4: Rigorous Benchmark Suite'));
  assert.ok(readme.includes('Stage 5: Nsight Profiling'));
  assert.ok(readme.includes('Timing Protocol: 25 warm-up iterations, 100 synchronized timed iterations'));
});
