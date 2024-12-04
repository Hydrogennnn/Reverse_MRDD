import torch
import torch.nn as nn

a = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float)

b = torch.tensor([[3, 5],
                  [8, 6]], dtype=torch.float)

loss_fn1 = torch.nn.MSELoss(reduction='none')
loss1 = loss_fn1(a.float(), b.float())
print(loss1)  # 输出结果：tensor([[ 4.,  9.],
#                 [25.,  4.]])

loss_fn2 = torch.nn.MSELoss(reduction='sum')
loss2 = loss_fn2(a.float(), b.float())
print(loss2)  # 输出结果：tensor(42.)

loss_fn3 = torch.nn.MSELoss(reduction='mean')
loss3 = loss_fn3(a.float(), b.float())
print(loss3)  # 输出结果：tensor(10.5000)
