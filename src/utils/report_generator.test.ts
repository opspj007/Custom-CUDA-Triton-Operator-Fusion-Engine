import test from 'node:test';
import assert from 'node:assert/strict';
import { generateReportHtml } from './report_generator';
import { calculateTheoreticalProfile, KernelExecutionConfig } from '../core/gpu_simulation';

test('generateReportHtml creates valid PDF-ready HTML containing comparative table and bottleneck analysis', () => {
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
  const html = generateReportHtml(
    config,
    profile.eagerResult,
    profile.inductorResult,
    profile.tritonResult,
    profile.cudaResult
  );

  // Check document structure
  assert.ok(html.includes('<!DOCTYPE html>'), 'Contains DOCTYPE html');
  assert.ok(html.includes('@page {'), 'Contains print @page styling');
  assert.ok(html.includes('window.print()'), 'Contains PDF print trigger');

  // Check comparative latency table
  assert.ok(html.includes('1. Comparative Performance Matrix'), 'Contains comparative matrix header');
  assert.ok(html.includes('PyTorch Eager (ATen)'), 'Contains PyTorch Eager in table');
  assert.ok(html.includes('Custom OpenAI Triton Kernel'), 'Contains Triton row in table');
  assert.ok(html.includes('Custom CUDA C++ (__shfl_down_sync)'), 'Contains CUDA row in table');

  // Check bottleneck analysis
  assert.ok(html.includes('2. Summary of Bottleneck Analysis'), 'Contains bottleneck analysis section');
  assert.ok(html.includes('Long Scoreboard (DRAM latency)'), 'Mentions Long Scoreboard stall bottleneck');
  assert.ok(html.includes('aten::add'), 'Mentions aten::add');
  assert.ok(html.includes('aten::layer_norm'), 'Mentions aten::layer_norm');
  assert.ok(html.includes('aten::silu'), 'Mentions aten::silu');
  assert.ok(html.includes('__shfl_down_sync'), 'Mentions __shfl_down_sync register shuffle solution');

  // Check hardware specs
  assert.ok(html.includes('NVIDIA GeForce RTX 4090'), 'Contains GPU model name');
  assert.ok(html.includes('1008 GB/s'), 'Contains memory bandwidth');
});
