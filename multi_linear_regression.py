import torch
import torch.nn as nn

torch.manual_seed(1)

# Initialization
## Training Dataset
x1 = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2 = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3 = torch.FloatTensor([[75], [93], [80], [100], [70]])
y = torch.FloatTensor([[152], [185], [180], [196], [142]])

## Weight & Bias
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

## Optimizer
optimizer = torch.optim.SGD([w1, w2, w3, b], lr=4e-5)

## Total number of epoch
total_ep = 5000

# Learning
for ep in range(total_ep+1):

    ## Hypothesis
    H = w1 * x1 + w2 * x2 + w3 * x3 + b

    ## Cost function
    mse_loss = nn.MSELoss()
    cost = mse_loss(y, H)

    ## Optimization
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    ## Result
    if ep % 10 == 0:
        print(H.size())
        print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, W3: {:.3f}, bias: {:.3f}, Cost: {:.6f}'.format(ep, total_ep, w1.item(), w2.item(), w3.item(), b.item(), cost.item()))


# Output
# Epoch 5000/5000 W1: 1.211, W2: 0.657, W3: 0.153, bias: 0.002, Cost: 0.112730