# Parallel Matrix Multiplication Lab

## Overview

This repository implements the same 4000 × 4000 matrix multiplication problem using four different computing models:

* **Sequential CPU** — single execution flow
* **OpenMP** — shared-memory CPU parallelism using multiple threads
* **MPI** — distributed-memory parallelism using multiple processes across nodes
* **CUDA** — GPU parallelism using a two-dimensional grid of CUDA threads

The purpose of the experiment is to study how the same computational problem behaves when the execution model changes.

The repository contains the implementation source code, concise experiment notes, execution commands, result screenshots, and the measured performance comparison.

---

## 1. Problem Definition

The same input is used for all four implementations.

```text
A = 4000 × 4000 matrix
B = 4000 × 4000 matrix
A[i][j] = 1.0
B[i][j] = 1.0

C = A × B
```

Matrix multiplication is computed as:

```text
C[i][j] = Σ A[i][k] × B[k][j]
```

For `C[0][0]`, there are 4000 terms and every term is:

```text
1.0 × 1.0 = 1.0
```

Therefore:

```text
C[0][0] = 4000.00
```

This value is used as the correctness verification for all four implementations.

---

## 2. Experiment Objective

The experiment compares four execution models for the same matrix multiplication workload:

```text
Sequential CPU
      ↓
OpenMP Shared Memory
      ↓
MPI Distributed Memory
      ↓
CUDA GPU Parallelism
      ↓
Performance and Correctness Comparison
```

The mathematical operation remains the same. The main difference is how the computation is divided and where it is executed.

---

## 3. Four Implementations

| Implementation | Execution Model           | Main Approach                                   | Configuration                 |
| -------------- | ------------------------- | ----------------------------------------------- | ----------------------------- |
| Sequential     | Single CPU execution flow | Three nested loops execute sequentially         | Single execution flow         |
| OpenMP         | Shared-memory CPU         | Outer-loop iterations distributed among threads | 8 OpenMP threads              |
| MPI            | Distributed memory        | Matrix rows divided among independent processes | 4 MPI processes / 4 VMs       |
| CUDA           | GPU parallelism           | Output elements computed by CUDA threads        | 16 × 16 block, 250 × 250 grid |

### Sequential

The sequential implementation performs the complete matrix multiplication using one CPU execution flow. It serves as the baseline for comparing the parallel implementations.

### OpenMP

The OpenMP implementation keeps the same matrix multiplication algorithm but distributes independent outer-loop iterations among multiple CPU threads. The experiment uses 8 OpenMP threads.

The main parallelization directive is:

```c
#pragma omp parallel for private(j, k)
```

Matrices `A`, `B`, and `C` remain in shared memory while different threads process different rows.

### MPI

The MPI implementation uses distributed memory. Four MPI processes participate in the computation, with each process handling 1000 rows.

```text
Rank 0 → 1000 rows
Rank 1 → 1000 rows
Rank 2 → 1000 rows
Rank 3 → 1000 rows
```

The major communication stages are:

```text
MPI_Scatter
    ↓
Distribute rows of A

MPI_Bcast
    ↓
Share matrix B

Local computation
    ↓
Each rank computes its portion of C

MPI_Gather
    ↓
Collect the complete matrix C on Rank 0
```

### CUDA

The CUDA implementation offloads the matrix multiplication to an NVIDIA GPU.

The execution configuration used in the experiment is:

```text
Matrix size = 4000 × 4000
Block size  = 16 × 16 threads
Grid size   = 250 × 250 blocks
```

Each CUDA thread determines its output `(row, column)` position and computes one element of matrix `C`.

---

## 4. Experiment Flow

![Experiment Flow](docs/experiment-flow.svg)

The experiment keeps the mathematical problem constant while changing the execution model from single CPU execution to shared-memory threads, distributed processes, and GPU threads.

---

# 5. Recorded Results

The following values are the measured results from the runs represented by the screenshots in this repository.

| Implementation | Configuration                 |              Recorded Time | Relative to Sequential | Verification |
| -------------- | ----------------------------- | -------------------------: | ---------------------: | -----------: |
| Sequential     | Single CPU execution flow     |           **339.308583 s** |                  1.00× |      4000.00 |
| OpenMP         | 8 CPU threads                 |           **107.938463 s** |                  3.14× |      4000.00 |
| MPI            | 4 processes / 4 VMs           |           **226.167575 s** |                  1.50× |      4000.00 |
| CUDA           | 16 × 16 block, 250 × 250 grid | **0.211245 s** kernel time |              1606.23×* |      4000.00 |

* The CUDA value shown here is **kernel execution time**. It should not be interpreted as a directly equivalent whole-program timing because host-to-device and device-to-host transfers are measured separately by the CUDA program.

### Speedup Calculation

For the CPU and MPI measurements:

```text
Speedup = Sequential Time / Implementation Time
```

Using the recorded values:

```text
OpenMP = 339.308583 / 107.938463 ≈ 3.14×
MPI    = 339.308583 / 226.167575 ≈ 1.50×
```

The CUDA kernel-only comparison gives:

```text
CUDA kernel = 339.308583 / 0.211245 ≈ 1606.23×
```

This CUDA comparison is reported separately because the measured quantity is kernel execution time rather than the complete CUDA phase.

---

# 6. Why the Results Differ

The algorithms perform the same mathematical operation, but the execution models introduce different ways of distributing the work.

### Sequential

Only one execution flow performs the computation. There is no explicit parallel execution, so the measured time forms the baseline.

### OpenMP

Multiple CPU threads execute independent loop iterations concurrently while sharing the same memory. This reduces the elapsed computation time compared with the sequential baseline.

### MPI

The computation is divided among independent processes running across multiple virtual machines. MPI also requires communication to distribute and collect data. In this experiment, that communication occurs through the VM network.

### CUDA

The computation is transferred to the GPU, where a large number of logical threads execute the output-element calculations concurrently. The measured kernel time therefore reflects GPU computation rather than the complete CPU-to-GPU workflow.

---

# 7. Execution

## 7.1 Sequential

Open a terminal and enter the Sequential directory:

```bash
cd sequential
```

Compile:

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
```

Run:

```bash
./matrix_sequential
```

Expected verification:

```text
Verification C[0][0] = 4000.00
```

---

## 7.2 OpenMP

Enter the OpenMP directory:

```bash
cd ../openmp
```

Set the number of OpenMP threads:

```bash
export OMP_NUM_THREADS=8
```

Compile:

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

Run:

```bash
./matrix_openmp
```

Expected verification:

```text
Verification C[0][0] = 4000.00
```

---

## 7.3 MPI

The MPI experiment uses one Master node and three Worker nodes.

The hostfile template is provided as:

```text
mpi/hosts.example
```

Create the working hostfile:

```bash
cd ../mpi
cp hosts.example hosts
```

The hostfile follows the four-node structure used in the experiment:

```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

Compile on the Master node:

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

Copy the executable to the Worker nodes:

```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

Run four MPI processes:

```bash
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

Expected verification:

```text
Verification C[0][0] = 4000.00
```

Each process handles:

```text
1000 rows
```

The MPI implementation uses `MPI_Scatter`, `MPI_Bcast`, local computation, and `MPI_Gather`.

---

## 7.4 CUDA

Enter the CUDA directory:

```bash
cd ../cuda
```

Compile the CUDA source using `nvcc`:

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

Run:

```bash
./matrix_cuda
```

The program reports the grid size, block size, kernel execution time, total CUDA phase time, and verification result.

Expected verification:

```text
Verification C[0][0] = 4000.00
```

---

# 8. Result Evidence

## Sequential

![Sequential Result](sequential/results/seq-output1.jpeg)

## OpenMP

![OpenMP Result](openmp/results/openmp-output.png)

## MPI

![MPI Result](mpi/results/final_output.png)

## CUDA

![CUDA Result](cuda/results/final_output.png)

---

# 9. Repository Structure

```text
parallel-matrix-multiplication/
│
├── README.md
├── .gitignore
│
├── sequential/
│   ├── README.md
│   ├── matrix_sequential.c
│   └── results/
│       └── final_output.jpeg
│
├── openmp/
│   ├── README.md
│   ├── matrix_openmp.c
│   └── results/
│       └── final_output.png
│
├── mpi/
│   ├── README.md
│   ├── matrix_mpi.c
│   ├── hosts.example
│   └── results/
│       └── final_output.png
│
├── cuda/
│   ├── README.md
│   ├── matrix_cuda.cu
│   └── results/
│       └── final_output.png
│
└── docs/
    └── experiment-flow.svg
```

---

# 10. Verification

All four implementations produce the same correctness value:

```text
C[0][0] = 4000.00
```

This demonstrates that the four implementations are computing the same matrix multiplication result while using different execution models.

Correctness is therefore checked independently from performance.

---

# 11. Key Observations

The experiment demonstrates four different approaches to executing the same computational workload:

```text
Sequential
    → single CPU execution flow

OpenMP
    → multiple CPU threads sharing memory

MPI
    → multiple processes using distributed memory

CUDA
    → GPU threads executing the computation in parallel
```

The measured results also show that performance depends not only on the number of processing units, but on the execution model, communication overhead, memory behavior, virtualization overhead, and what portion of the computation is included in the timing.

---

# 12. Conclusion

This experiment demonstrates how a single matrix multiplication problem can be implemented using four distinct computing models: sequential CPU execution, OpenMP shared-memory parallelism, MPI distributed-memory parallelism, and CUDA GPU parallelism.

The sequential implementation establishes the baseline execution time. OpenMP improves the CPU execution by distributing independent iterations among multiple threads. MPI extends the computation across separate processes and virtual machines, while also introducing communication overhead. CUDA maps the matrix computation onto GPU threads and reports a separate kernel execution time.

Although the execution strategies are different, all four implementations produce the expected verification value:

```text
C[0][0] = 4000.00
```

Therefore, the experiment demonstrates both **correctness preservation** and the effect of different parallel computing architectures on execution time. The measured timings should be interpreted as results of this specific laboratory environment and run rather than as universal performance values.

---

## Source Basis

The implementations and experiment organization follow the supplied laboratory manual covering:

* 4000 × 4000 matrix multiplication
* Sequential CPU execution
* OpenMP shared-memory execution
* MPI execution across four processes/nodes
* CUDA GPU execution
* Correctness verification using `C[0][0] = 4000.00`
* Performance comparison across the four models
