import numpy as np
import torch
import loss as CL

from loss import ContrastiveLoss


def train(model, optimizer, device, data):
    loss = CL.ContrastiveLoss(10e-4, 0.75, 1).to(device)

    model.train()
    # batch implementalasa

    i = 0
    print(len(data[0]))
    for pair in data:
        s1 = torch.from_numpy(pair[0]).float().to(device)
        s2 = torch.from_numpy(pair[1]).float().to(device)
        y = torch.ShortTensor(pair[2]).to(device)

        # optimizeer.zero grad
        s1, s2 = model(s1, s2)
        loss.forward(s1, s2, y)
        loss.backward()
        optimizer.step()
        print(i)
        i += 1


def eval(model, loss, data):
    return
