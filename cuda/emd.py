import importlib
import os

import torch
from torch import nn


earthMover_found = importlib.find_loader("earthMoverearthMover_3D") is not None
if not earthMover_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting EarthMover 3D")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    build_path = cur_path.replace('cuda', 'tmp')
    os.makedirs(build_path, exist_ok=True)

    from torch.utils.cpp_extension import load
    emd_cuda = load(name="emd_cuda",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["emd.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["emd_kernel.cu"]),
              ], build_directory=build_path)
    print("Loaded JIT 3D CUDA earth mover distance")

else:
    import emd_cuda
    print("Loaded compiled 3D CUDA earth mover distance")

class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


class earth_mover_distance(nn.Module):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, n, 3)
        xyz2 (torch.Tensor): (b, n, 3)

    Returns:
        cost (torch.Tensor): (b)

    """
    def __init__(self) -> None:
        super(earth_mover_distance, self).__init__()

    def forward(self, xyz1, xyz2):
        _, _, dim = xyz1.size()
        assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
        _, _, dim = xyz2.size()
        assert dim==3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
        cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
        return cost

