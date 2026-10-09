import numpy as np
from numba import cuda


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]
        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.shape[0]
    threads = 256
    blocks = (N + threads - 1) // threads
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)
    stencil_1d[blocks, threads](d_in, d_out, N)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


if __name__ == "__main__":
    N = 100_007
    h_in = np.random.rand(N).astype(np.float32)
    h_gpu = run_stencil(h_in)
    h_cpu = cpu_stencil(h_in)
    assert np.allclose(h_gpu, h_cpu, atol=1e-4)
    print(f"TASK 2 PASSED: MAX DELTA = {np.max(np.abs(h_gpu - h_cpu))}")
