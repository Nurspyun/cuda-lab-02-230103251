import numpy as np
from numba import cuda


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]
    threads_per_block = 256
    blocks_per_grid = 64
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[blocks_per_grid, threads_per_block](d_arr, factor, N)
    cuda.synchronize()
    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 2 ** 24
    factor = 3.5
    res = run_grid_stride(np.ones(N, dtype=np.float32), factor)
    ok = bool(np.all(res == factor))
    print(f"Threads launched: {256 * 64:,} for N = {N:,}")
    print(f"All {N:,} elements equal {factor}: {ok}")
    assert ok
    print("TASK 3 PASSED")
