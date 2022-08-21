import torch
import torch.nn as nn

torch.manual_seed(1)

# Initialization
## Training Dataset
X = torch.FloatTensor([[73, 80, 75],[93, 88, 93],[89, 91, 80],[96, 98, 100],[73, 66, 70]])
y = torch.FloatTensor([[152], [185], [180], [196], [142]])

## Weight & Bias
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

## Optimizer
optimizer = torch.optim.SGD([W, b], lr=4e-5)

## Total number of epoch
total_ep = 5000

# Learning
for ep in range(total_ep+1):

    ## Hypothesis
    H = X.matmul(W) + b

    ## Cost function
    cost = nn.MSELoss()(y, H)

    ## Optimization
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    ## Result
    if ep % 10 == 0:
        print('Epoch {:4d}/{} Weights: {}, bias: {:.3f}, Cost: {:.6f}'.format(ep, total_ep, W.squeeze().detach(), b.item(), cost.item()))


# Output
# Epoch 5000/5000 Weights: tensor([1.2112, 0.6568, 0.1526]), bias: 0.002, Cost: 0.112725