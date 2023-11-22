# PyTorch Wrapper for Point-cloud Earth-Mover-Distance (EMD)

## Dependency

The code has been tested on Ubuntu 22.04, PyTorch 2.0.0, CUDA 11.2.

## Usage

```py
from emd import earth_mover_distance

d = earth_mover_distance(p1, p2, transpose=False)  # p1: B x N1 x 3, p2: B x N2 x 3

import torch, emd.earth_mover_distance
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
```
[ ] 修改readme
## Author

The cuda code is originally written by Haoqiang Fan. The PyTorch wrapper is written by Kaichun Mo. Also, Jiayuan Gu provided helps.

## License

MIT

