from torch import nn
import numpy as np
import torch

class CrossEntropyLoss2d(nn.Module):
  def __init__(self):
    super(CrossEntropyLoss2d, self).__init__()
    base_loss = nn.CrossEntropyLoss(weight=None,
                                    size_average=False,
                                    reduce=False)
    if torch.cuda.is_available():
      self.ce_loss = base_loss.cuda()
    else:
      self.ce_loss = base_loss
    # self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
    #                                    size_average=False, reduce=False)

  def forward(self, inputs_scales, targets_scales):
    inputs = inputs_scales[0]
    a, b, c, d = targets_scales.shape
    targets = targets_scales.view(a,c,d) # YARG
    mask = targets > 0
    targets_m = targets.clone()
    targets_m[mask] -= 1
    loss_all = self.ce_loss(inputs, targets_m.long())
    return torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())

def loss_function():
  return CrossEntropyLoss2d()
