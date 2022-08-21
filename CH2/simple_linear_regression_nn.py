from statistics import mode
import torch
import torch.nn as nn

torch.manual_seed(1)

# Initialization
## Training Dataset
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

## Hypothesis = Linear Regression
model = nn.Linear(1, 1)

## Optimizer = SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

## Total number of epoch
total_ep = 2000


# Learning
for ep in range(total_ep+1):

    ## H = W * x_train + b
    H = model(x_train)

    ## Cost function = MSE
    cost = nn.MSELoss()(y_train, H)

    ## Optimization
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    ## Result
    if ep % 10 == 0:
        print('Epoch {:4d}/{} , Cost: {:.6f}'.format(ep, total_ep, cost.item()))
        print(list(model.parameters()))
        print('--------------------------------')
        ### print(list(model.parameters())) => Linear Regression Model의 parameter들을 출력해준다. Weight(W) / Bias(b) 순서.
        ### list()를 잊지 말자.
    
x_test = torch.FloatTensor([[4]])
y_test = model(x_test)

print('\n<After Learning>')
print('Input : 4, Output : ', y_test)
