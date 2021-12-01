import torch
import loss as CL


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

        optimizer.zero_grad()
        s1, s2 = model(s1, s2)
        result = loss(s1, s2, y)
        result.backward()
        optimizer.step()
        print(i)
        i += 1


@torch.no_grad()
def eval(model, loss, data, device):
    model.eval()

    for pair in data:
        s1 = torch.from_numpy(pair[0]).float().to(device)
        s2 = torch.from_numpy(pair[1]).float().to(device)
        y = torch.ShortTensor(pair[2]).to(device)

        s1, s2 = model(s1, s2)
        result = loss(s1,s2,y)

    #TODO calculate accuracy
    return
