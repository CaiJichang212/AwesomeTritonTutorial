import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # pointer to first vector
    y_ptr,  # pointer to second vector
    output_ptr,  # pointer to output vector
    n_elements,  # size of the vector
    BLOCK_SIZE: tl.constexpr,  # block size
):
    # This thread block will process BLOCK_SIZE elements
    pid = tl.program_id(axis=0)  # 1D grid
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load data from memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store back to memory
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Output tensor
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    
    # Grid function: how many blocks we need
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# Test
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

if torch.allclose(output_torch, output_triton):
    print('✅ Triton and Torch match!')
else:
    print('❌ Triton and Torch differ!')
