import torch
import torch.nn as nn

# Initialization
## Training Dataset
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

## Weight & Bias
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

## Optimizer = SGD
optimizer = torch.optim.SGD([W, b], lr=0.01)

## Total number of epoch
num_epoch = 2000


# Learning
for epoch in range(num_epoch+1):

    ## Hypothesis = Linear Regression
    H = W * x_train + b

    ## Cost function = MSE
    mse_loss = nn.MSELoss()
    cost = mse_loss(y_train, H)

    ## Optimization
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    ## Result
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Weight: {:.3f}, bias: {:.3f} Cost: {:.6f}'.format(epoch, num_epoch, W.item(), b.item(), cost.item()))
        ### 문자열 포맷팅 (String Formatting) : 여러 값들을 입력 받아 String 안에 순서에 맞게 넣어준다. {}와 .format 함수를 같이 쓴다.
        ### item() 함수 : 딕셔너리 (Dictionary)안에 있는 원소 값을 불러옴.