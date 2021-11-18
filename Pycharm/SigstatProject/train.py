import numpy as np
import torch


def train(model, optimizer, loss, device, data):
    model.train()
    # batch implementalasa

    for pair in data[0]:
        print(pair)
        s1 = torch.from_numpy(pair[0]).float().to(device)
        s2 = torch.from_numpy(pair[1]).float().to(device)
        y = torch.from_numpy(pair[2]).float().to(device)


        # optimizeer.zero grad
        s1, s2 = model(s1, s2)
        loss = loss(s1, s2, y)
        loss.backward()
        optimizer.step()


def eval(model, loss, data):
    return
