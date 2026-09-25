# Results Comparison — Sequential vs OpenMP

Both implementations use the same `4000 × 4000` matrix multiplication and verify the result using `C[0][0] = 4000.00`.

## Recorded Results

| Implementation | Execution Time | Threads | Verification | Speedup |
|---|---:|---:|---:|---:|
| Sequential | 339.308583 s | 1 execution flow | 4000.00 | 1.00× |
| OpenMP | 107.938463 s | 8 | 4000.00 | 3.14× |

## Calculation

```text
Speedup = Sequential Time / OpenMP Time
        = 339.308583 / 107.938463
        ≈ 3.14×
```

## Interpretation

The OpenMP implementation reduces execution time by dividing independent outer-loop iterations among multiple shared-memory CPU threads. The correctness value remains `4000.00`, so the performance comparison uses the same mathematical workload and the same verification condition.

The recorded timing is machine- and run-dependent; this repository stores the values shown in the submitted terminal outputs.
