This is a part of Jacob Hilton's Deep Learning Curriculum: https://github.com/jacobhilton/deep_learning_curriculum/blob/master/3-Training-at-Scale.md

Notes:
- fused adam runs ~3x faster than naive PyTorch adam on a P4000 GPU over 1 linear layer. 
- for some reason pytorch's fused adam runs slower when fused=True? need to test this in a more realistic model setting. 
- (part of this is clearly because of forking functionality)
- I believe the core speedup is due to the computations all occurring on chip instead of being copied back and forth. The elementwise operations themselves are easily fused by PyTorch's JIT 
- have not yet experimented with block size tuning.

Resources:
https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/triton.py

