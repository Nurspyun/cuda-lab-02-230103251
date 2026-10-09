%%writefile README.md
# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103251
**Allocated GPU Node:** Tesla T4
**CUDA Compute Capability:** 7.5
**Official Verification Token:** D6B02C8BB195EACA48DA

## Task 1: Warp Divergence Benchmark

N = 2^20 elements (float32), 1,000 iterations per element, 256 threads per block.
Kernel-only time, averaged over 10 trials after one warm-up launch.

| Kernel | Avg time (ms) | Slowdown vs A |
|---|---|---|
| A (Uniform) | 20.5518 | 1.00x |
| B (Full Divergence) | 95.8318 | 4.66x |
| C (Warp-Aligned) | 46.4212 | 2.26x |

**Analysis**

- Kernel B is the slowest. Even and odd threads of the same warp take different branches, so the hardware executes both paths one after the other and every warp pays for both.
- Kernel C is faster than B because whole warps take one branch, so there is no divergence inside a warp.
- Kernel C is still slower than A. The second path (subtract and divide) is more expensive than the first (multiply-add), so half of the warps in C do heavier work than every warp in A. That part of the gap is not caused by divergence.

## Task 2: 1D Stencil

N = 100,007. Output: TASK 2 PASSED: MAX DELTA = 5.960464477539063e-08

## Task 3: Grid-Stride Loop

N = 16,777,216 elements with 256 x 64 = 16,384 threads. All elements matched the factor.

## Task 4: Sobel-X (2048 x 2048)

Output shape (2048, 2048). Max delta vs NumPy reference: 4.76837158203125e-07.

## Verification

verify_submission.py: all three checks passed.
