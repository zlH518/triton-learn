import copy
import torch
import triton
import triton.language as tl

'''
为了减少navie softmax中的读写操作，用于内核融合
'''

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def softmax_kernel_oneline(x_ptr, output_ptr, x_stride, n_clos, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = x_ptr + pid * x_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_clos
    input = tl.load(start + offsets, mask=mask, other=-float('inf'))
    input = input - tl.max(input)
    input = tl.exp(input)
    row_sum = tl.sum(input, axis=0)
    output = input / row_sum
    output_start = output_ptr + pid * x_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_clos
    tl.store(output_start + offsets, output, mask=mask)

@triton.jit
def softmax_kernel_step(x_ptr, output_ptr, x_stride, n_rows, n_clos, num_stages: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(axis=0)
    step =  tl.num_programs(axis=0)

    for row in tl.range(row_start, n_rows, step, num_stages=num_stages):
        start = x_ptr + row * x_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_clos
        input = tl.load(start + offsets, mask=mask, other=-float('inf'))
        input = input - tl.max(input, axis=0)
        input = tl.exp(input)
        row_sum = tl.sum(input, axis=0)
        output = input / row_sum
        output_start = output_ptr + row * x_stride
        tl.store(output_start + offsets, output, mask=mask)


def softmax_triton_step(x):
    n_rows, n_clos = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_clos)
    grid = (min(n_rows, 256),)
    output = torch.zeros_like(x)
    num_stages = 4

    softmax_kernel_step[grid](x, output, x.stride(0), n_rows, n_clos, num_stages=num_stages, BLOCK_SIZE=BLOCK_SIZE)

    return output


def softmax_triton_oneline(x):
    n_rows, n_clos = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_clos)
    grid = (n_rows,)
    output = torch.zeros_like(x)

    softmax_kernel_oneline[grid](x, output, x.stride(0), n_clos, BLOCK_SIZE=BLOCK_SIZE)

    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton_oneline', 'triton_step', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton_oneline", "Triton_step", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('yellow', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 16384},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton_oneline':
        ms = triton.testing.do_bench(lambda: softmax_triton_oneline(x))
    if provider == 'triton_step':
        ms = triton.testing.do_bench(lambda: softmax_triton_step(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: softmax_navie(x, dim=1))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


def softmax_navie(x, dim):
    '''
    output = e^x / sum(e^x)
    '''
    x = x - x.max()
    z = torch.exp(x)
    a = z.sum(dim=dim)[:,None]
    output = z / a
    return output



def main():
    torch.manual_seed(1427)
    M = 1024
    N = 1024
    x = torch.randint(low=0, high=10, size=(M, N), device="cuda").float()
    print(x.numel(), x.element_size())
    print(f'x: {x}')
    output = torch.softmax(x, dim=1)
    print(f'output: {output}')
    output_navie = softmax_navie(x, dim=1)
    print(f'output_navie: {output_navie}')
    output_triton_oneline = softmax_triton_oneline(x)
    print(f'output_triton_oneline: {output_triton_oneline}')
    output_triton_step = softmax_triton_step(x)
    print(f'output_triton_step: {output_triton_step}')

    benchmark.run(show_plots=True, print_data=True, save_path='./output')



if __name__ == "__main__":
    main()
