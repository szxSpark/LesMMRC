# import torch
# import torch.nn.functional as F
# import torch.nn as nn
#
# def custom_loss_fn(data, labels):
#     loss = torch.zeros(1)
#     for d, label in zip(data, labels):
#         loss -= torch.log(d[label]).cpu()
#
#     loss /= data.size(0)
#     return loss
#
# loss = nn.NLLLoss()
# loss1 = nn.CrossEntropyLoss()
#
# m = nn.Softmax()
# m1 = nn.LogSoftmax()
# x = torch.randn(4, 3, requires_grad=True)
# y = torch.empty(4, dtype=torch.long).random_(3)
#
# r = loss(m1(x), y)
# print(r)
#
# r = loss1(x, y)  # 交叉shang，logSoftmax + NLL
# print(r)
#
# r = custom_loss_fn(m(x), y)  # open-source, softmax后 加上负对数
# print(r)
#
# a = torch.randn(6, 2)
# print(a)
# print(a.transpose(0, 1)[0])
# print(a.transpose(0, 1)[1])