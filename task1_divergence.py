import time
import numpy as np
from numba import cuda

N = 2 ** 20
THREADS = 256


@cuda.jit
def kernel_a(y):
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        v = y[idx]
        for _ in range(1000):
            v = v * 1.0001 + 0.0001
        y[idx] = v


@cuda.jit
def kernel_b(y):
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        v = y[idx]
        if idx % 2 == 0:
            for _ in range(1000):
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(1000):
                v = (v - 0.0001) / 1.0001
        y[idx] = v


@cuda.jit
def kernel_c(y):
    idx = cuda.grid(1)
    if idx < y.shape[0]:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(1000):
                v = v * 1.0001 + 0.0001
        else:
            for _ in range(1000):
                v = (v - 0.0001) / 1.0001
        y[idx] = v


def bench(kernel, d_y, blocks):
    kernel[blocks, THREADS](d_y)  # warm-up
    cuda.synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        kernel[blocks, THREADS](d_y)
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main():
    blocks = (N + THREADS - 1) // THREADS
    h = np.random.rand(N).astype(np.float32)
    d_y = cuda.to_device(h)
    results = {}
    for name, k in (("A (Uniform)", kernel_a),
                    ("B (Full Divergence)", kernel_b),
                    ("C (Warp-Aligned)", kernel_c)):
        results[name] = bench(k, d_y, blocks)
    base = results["A (Uniform)"]
    print("| Kernel | Avg time (ms) | Slowdown vs A |")
    print("|---|---|---|")
    for name, t in results.items():
        print(f"| {name} | {t:.4f} | {t / base:.2f}x |")


if __name__ == "__main__":
    main()
