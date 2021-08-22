from torch import nn

def crossEntropyLoss(reduction='sum'):
  return nn.CrossEntropyLoss(reduction=reduction)