import torch
import torch.nn as nn
img = torch.randn(1,3,4,4,requires_grad = True)
label = torch.tensor([[[0,0,0,0],
                       [0,1,1,0],
                       [0,1,1,0],
                       [0,0,0,0]]])
model = nn.Sequential(nn.Conv2d(3,8,3,1,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(8,8,3,1,padding = 1),
                      nn.ReLU(),
                      nn.Conv2d(8,2,3,1,padding = 1 ))

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_f = nn.CrossEntropyLoss()
for i in range(20):
    out = model(img)
    out = nn.Softmax(dim = 1)(out)
    loss = loss_f(out,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Loss:', loss)
    print('Output:', out)
