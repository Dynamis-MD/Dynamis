# Experiment 4: Communication Backend Comparison

## Backends

| # | Backend | Implementation | Mechanism |
|---|---------|---------------|-----------|
| 1 | **NCCL all_to_all** | `dist.all_to_all_single` | Collective, all ranks participate |
| 2 | **IPC one-sided push** | `dynamis_state.IPCExchanger` | CUDA IPC, sender writes directly to receiver memory |


## 4a. Micro-benchmark (Communication Only)

**Goal**: Isolate communication performance from compute. 
https://github.com/LuckyLittleMonster/benchmark.git

## 4b. End-to-End MD Performance

**Goal**: Measure actual steps/s improvement when replacing the communication backend in the full MD loop.

### Configuration

| System | Atoms | GPU counts | Backends |
|--------|------:|-----------|----------|
| 1AKI | 25,846 | 4 | NCCL, Push Only |
| 2RIK | 112,727 | 8, 16 | NCCL, Push Only |
| 9GMO | 1,536,442 | 16, 32 | NCCL, Push Only |

- Steps: 200, steady-state

### Metrics

| Metric | Definition |
|--------|-----------|
| Steps/s | Overall throughput |
| Comm time (ms/step) | Sum of all communication phases |
| Comm speedup | Comm_time(NCCL) / Comm_time(alternative) |
| End-to-end speedup | Steps/s(alternative) / Steps/s(NCCL) |

## Output

### Micro-benchmark figures
- Latency vs. message size curves (one subplot per sparsity level, lines = backends)
- Throughput vs. rank count at fixed message size
- Sparsity crossover point: where sparse P2P beats NCCL

### End-to-end figures
- Grouped bar chart: steps/s per system, bars = backends
- Communication time breakdown: stacked bar per backend
- Speedup annotation on bars
