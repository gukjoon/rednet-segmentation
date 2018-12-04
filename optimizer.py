import torch.optim as optim

def optimizer(network):
  return optim.SGD(network.parameters(), lr=0.001, momentum=0.9), None
