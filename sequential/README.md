# Experiment 1 — Sequential Matrix Multiplication

## Objective

Establish a performance baseline by multiplying two `4000 × 4000` matrices using a single sequential execution flow.

## Core Logic

```text
for each row i
    for each column j
        for each k
            C[i][j] += A[i][k] × B[k][j]
```

**Time complexity:** `O(N³)`

## Build & Run

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential
```

## Recorded Result

```text
Initializing 4000 x 4000 matrices...

Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 339.308583 seconds
Verification C[0][0] = 4000.00
```

| Metric | Value |
|---|---:|
| Matrix size | 4000 × 4000 |
| Execution time | **339.308583 s** |
| Verification | **4000.00** |
| Purpose | Baseline for parallel comparison |

## Why this result matters

The program completes the full `O(N³)` computation without parallel threads. The recorded **339.308583 s** becomes the reference point for measuring OpenMP speedup.

## Final Output

![Sequential final output](results/final_output.png)
