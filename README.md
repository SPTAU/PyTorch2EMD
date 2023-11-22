# PyTorch Wrapper for Point-cloud Earth-Mover-Distance (EMD)

## Dependency

The code has been tested on Ubuntu 22.04, PyTorch 2.0.0, CUDA 11.2.

## Usage

```py
import torch
from cuda.emd import earth_mover_distance

emdLoss = earth_mover_distance()
points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
dist = emdLoss(points1, points2)
```
## Author

The cuda code is originally written by Haoqiang Fan. The PyTorch wrapper is written by Kaichun Mo. Also, Jiayuan Gu provided helps.

## License

MIT

