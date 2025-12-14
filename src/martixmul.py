import torch
import triton
import triton.language as tl

# 设置 PyTorch 打印精度为最高（全精度）
torch.set_printoptions(precision=10, sci_mode=False)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, output_ptr,
    # debug 输出 tensor
    debug_x_block_ptr, debug_y_block_ptr, debug_acc_ptr,
    M, N, K,
    x_stride_0, x_stride_1,
    y_stride_0, y_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    '''
    总的block的数量是M/block_size_m * N/block_size_n, 
    先找出当前block是处理的x的哪些行(start_row:start_row+block_size_m)，y的哪些列(start_clo:start_clo+block_size_n)
    然后对应的output的内存位置应该是(m:m+block_size_m,n:n+block_size_n)
    然后外层是个循环0-K，step是block_size_k
    取出x的(start_row:start_row+block_size_m, k:k+block_size_k)和y的(k:k+block_size_k, start_clo:start_clo+block_size_n)的值做dot
    累加到output对应的位置上即可
    '''
    # my version
    # 当前是第几个block
    pid = tl.program_id(axis=0)

    #将一维的pid定位转化为二维的pid
    # 是x row方向上第几个
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // (num_pid_n)
    pid_n = pid % (num_pid_n)

    # 计算当前block对应各个维度的偏移列表
    offsets_m = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offsets_n = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, K, BLOCK_SIZE_K):
        offsets_k = tl.arange(0, BLOCK_SIZE_K) + k

        x_ptrs = x_ptr + offsets_m[:, None] * x_stride_0 + offsets_k[None, :] * x_stride_1
        y_ptrs = y_ptr + offsets_k[:, None] * y_stride_0 + offsets_n[None, :] * y_stride_1

        x_mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        y_mask = (offsets_k[:, None] < K) & (offsets_n[None, :] < N)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y_block = tl.load(y_ptrs, mask=y_mask, other=0.0)

        acc = tl.dot(x_block, y_block, acc, input_precision="ieee")
        # acc = tl.dot(x_block, y_block, acc)
        
        # 只保存第一个 block 的第一次迭代结果到 debug tensor
        if pid == 0 and k == 0:
            # 计算 debug tensor 的偏移
            debug_offsets_m = tl.arange(0, BLOCK_SIZE_M)
            debug_offsets_k = tl.arange(0, BLOCK_SIZE_K)
            debug_offsets_n = tl.arange(0, BLOCK_SIZE_N)
            
            # 保存 x_block (BLOCK_SIZE_M, BLOCK_SIZE_K)
            x_block_ptrs = debug_x_block_ptr + debug_offsets_m[:, None] * BLOCK_SIZE_K + debug_offsets_k[None, :]
            tl.store(x_block_ptrs, x_block)
            
            # 保存 y_block (BLOCK_SIZE_K, BLOCK_SIZE_N)
            y_block_ptrs = debug_y_block_ptr + debug_offsets_k[:, None] * BLOCK_SIZE_N + debug_offsets_n[None, :]
            tl.store(y_block_ptrs, y_block)
            
            # 保存 acc (BLOCK_SIZE_M, BLOCK_SIZE_N)
            acc_ptrs = debug_acc_ptr + debug_offsets_m[:, None] * BLOCK_SIZE_N + debug_offsets_n[None, :]
            tl.store(acc_ptrs, acc)

    output_offsets = offsets_m[:, None] * output_stride_0 + offsets_n[None, :] * output_stride_1
    output_ptrs = output_ptr + output_offsets
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=mask)


def matmul_triton(x, y):
    assert x.shape[1] == y.shape[0], "矩阵维度不匹配"
    assert x.is_cuda and y.is_cuda, "输入必须在GPU上"
    
    M, K = x.shape
    K, N = y.shape
    
    output = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # 创建 debug tensor 用于保存中间结果
    debug_x_block = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_K), device=x.device, dtype=torch.float32)
    debug_y_block = torch.empty((BLOCK_SIZE_K, BLOCK_SIZE_N), device=x.device, dtype=torch.float32)
    debug_acc = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N), device=x.device, dtype=torch.float32)
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    matmul_kernel[grid](
        x, y, output,
        debug_x_block, debug_y_block, debug_acc,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # # 打印 debug 结果（最高精度）
    # print("========== DEBUG: Block 0, k=0 ==========")
    # print(f"x_block (from kernel):\n{debug_x_block}")
    # print(f"y_block (from kernel):\n{debug_y_block}")
    # print(f"acc after first dot:\n{debug_acc}")
    # print("==========================================")
    
    return output


def naive_matmul(x, y):
    assert x.shape[1] == y.shape[0]
    M, K, N = x.shape[0], x.shape[1], y.shape[1]
    output = torch.zeros((M, N), device="cuda")
    for i in range(M):
        for j in range(N):
            output[i, j] = sum(x[i, k]*y[k, j] for k in range(K))
        if i==0 and j==0:
            print("naive:", output[i,:])
    return output


def calaute_tile_address():
    M = 4
    N = 6
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_N = 2
    START_M = 1
    START_N = 3
    STRIDE = [N, 1]

    x = torch.randint(0, 6, (M, N), device="cuda")
    x_offsets = torch.arange(START_M, START_M+BLOCK_SIZE_M)
    y_offsets = torch.arange(START_N, START_N+BLOCK_SIZE_N)
    print((x_offsets*STRIDE[0])[:, None])
    print((y_offsets*STRIDE[1])[None, :])
    address = (x_offsets*STRIDE[0])[:, None] + (y_offsets*STRIDE[1])[None, :]
    print(address)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['K'],  # argument names to use as an x-axis for the plot
        x_vals=[4 * i for i in range(2, 10)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch', 'triton'],  # possible values for `line_arg``
        line_names=["Torch", "Triton"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 128, 'N': 128},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, K, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    y = torch.randn(N, K, device=DEVICE, dtype=torch.float32)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(x, y))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: matmul_triton(x, y))
    if provider == 'naive':
        ms = triton.testing.do_bench(lambda: naive_matmul(x, y))
    return ms




def main():
    torch.manual_seed(1407)
    M, K, N = 16, 16, 16
    # x = torch.randint(0, 6, (M, K), device="cuda", dtype=torch.float32)
    # y = torch.randint(0, 6, (K, N), device="cuda", dtype=torch.float32)
    x = torch.randn((M, K), device="cuda", dtype=torch.float32)
    y = torch.randn(K, N, device="cuda", dtype=torch.float32)
    print("x:", x)
    print("y:", y)
    output_torch = torch.matmul(x, y)
    print(f'output_torch: {output_torch}')

    output_naive = naive_matmul(x, y)
    print(f'output_naive: {output_naive}')
    
    output_triton = matmul_triton(x, y)
    print(f'output_triton: {output_triton}')
    
    assert torch.allclose(output_torch, output_triton)
    assert torch.allclose(output_torch, output_naive)
    benchmark.run(show_plots=True, print_data=True, save_path='./output')


if __name__ == "__main__":
    main()
