import triton
import torch

import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def dropout_kernel(
    x,
    keep,
    output,
    element_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_size
    x_data = tl.load(x + offsets, mask=mask)
    mask_data = tl.load(keep + offsets, mask=mask)
    output_data = tl.where(mask_data, x_data, 0.0)
    tl.store(output+offsets, output_data, mask)


@triton.jit
def seeded_dropout_kernel(
    x,
    p,
    output,
    seed,
    element_size,
    keep_mask_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_size
    x_data = tl.load(x + offsets, mask=mask)
    keep_mask_random = tl.rand(seed, offsets)
    keep_mask = keep_mask_random > p
    output_data = tl.where(keep_mask, x_data, 0.0)
    tl.store(output+offsets,output_data, mask)
    tl.store(keep_mask_ptr+offsets, keep_mask, mask)


def seeded_dropout(x, p, seed):
    assert x.is_contiguous()
    element_size = x.numel()
    keep_mask = torch.empty_like(x, device=DEVICE, dtype=torch.bool)
    output = torch.empty_like(x, device=DEVICE)
    grid = lambda meta: (triton.cdiv(element_size, meta["BLOCK_SIZE"]),)
    seeded_dropout_kernel[grid](x, p, output, seed, element_size, keep_mask, BLOCK_SIZE=1024)
    print(keep_mask)

    return output, keep_mask


def dropout(x, mask):
    assert x.is_contiguous()
    element_size = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(element_size, meta["BLOCK_SIZE"]),)
    dropout_kernel[grid](x, mask, output, element_size, BLOCK_SIZE=1024)

    return output

if __name__ == "__main__":
    seed = 1407
    torch.manual_seed(seed)
    shape = (4096,)
    p = 0.5
    x = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    mask = (torch.rand(shape, device=DEVICE) < p)
    output = torch.empty_like(x)

    print("navie:", torch.where(mask, x, 0.0))
    print("triton:", dropout(x, mask))
    output_data, mask_triton = seeded_dropout(x, p, seed)
    print("seeded_dropout:", output_data)

    print("mask torch:", mask)
    print("mask triton:", mask_triton)
    print((mask == mask_triton).to(torch.int32).sum())
