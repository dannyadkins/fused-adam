import torch 
import triton 
import triton.language as tl

@triton.jit 
def update_fn_kernel(p_ptr, grad_ptr, lr, beta1, beta2, eps, weight_decay, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets

    # load the slice of p 
    p = tl.load(offset_p_ptr, mask=mask)

    # load the slice of grad
    grad = tl.load(offset_grad_ptr, mask=mask)

    p = p - lr * grad

    # store the slice of p
    tl.store(offset_p_ptr, p, mask=mask)



def fused_update_fn(p, grad, lr, beta1, beta2, eps, weight_decay):
    # find the current program
    # calculate the offsets based on the block size (what is block size in this case? its 1? no it should be a big chunk of the params, needs to be power of 2)

    # here's a mental example: 
    # pid = 1
    # block_size = 128 
    # block_start = pid * block_size = 128
    # offsets = block_start + tl.arange(0, block_size) = 128 + [0, 1, 2, 3, ... 127] = [128, 129, 130, ... 255]
    # mask = offsets < num_params = [128, 129, 130, ... 255] < 256 = [True, True, True, ... True]. say num params is like 254. last 2 are false
    n_elements = p.numel()

    BLOCK_SIZE=128
    grid = triton.cdiv(n_elements, BLOCK_SIZE)

    # note: we do not need to malloc the output here because we are just doing a simple update in place 
    # p, grad are passed as ptrs because they are tensors? 
    # lr, beta1, beta2, eps, weight_decay are passed as values because they are scalars?
    update_fn_kernel[(grid,)](p, grad, lr, beta1, beta2, eps, weight_decay, n_elements, BLOCK_SIZE=BLOCK_SIZE)

class FusedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FusedAdam, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad = p.grad 
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                state = self.state[p]

                # TODO pass in momentum and anything else from state too here 
                fused_update_fn(
                    p, grad, lr, beta1, beta2, eps, weight_decay 
                )




if __name__ == "__main__":
    # initialize some random params and grads
    layer = torch.nn.Linear(256, 256, bias=False).cuda()
    params = list(layer.parameters())
    for p in params:
        p.grad = torch.randn_like(p)

    # adam = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    fused_adam = FusedAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    fused_adam.step()