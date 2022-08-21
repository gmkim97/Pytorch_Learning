import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# Initialization
## Training Dataset
X = torch.FloatTensor([[73, 80, 75],[93, 88, 93],[89, 91, 80],[96, 98, 100],[73, 66, 70]])
y = torch.FloatTensor([[152], [185], [180], [196], [142]])

## Hypothesis = Linear Regression
model = nn.Linear(3, 1)

## Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=4e-5)

## Total number of epoch
total_ep = 50

## Total data & Mini-batch
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
### Generally, batch_size is a multiply of 2

# Learning
for ep in range(total_ep+1):
    for batch_idx, samples in enumerate(dataloader):
    ### enumerate() : Input 안의 List의 원소와 인덱스가 Tuple 형태로 출력된다.
        
        X, y = samples
        print(samples)

        ## Hypothesis
        H = model(X)

        ## Cost function
        cost = func.mse_loss(y, H)
        ### This is same as cost = nn.MSELoss()(y_train, H)

        ## Optimization
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        ## Result
        print('Epoch {:4d}/{} , Batch {}/{} , Cost: {:.6f}'.format(ep, total_ep, batch_idx+1, len(dataloader), cost.item()))
        print('-----------------------')


# Output
# Total data = 5
# 3 iterations (Mini-batches) per a single epoch ==> # of weight(W) & bias(b) are updated
# Batch_size = 2 / 2 / 1