import torch
from loss_accuracy import accuracy

#train of the model
def train(model, optimizer, device, dataLoader, loss):
    model.train()
    #for loss and accuaracy
    calculated_loss = 0
    signs = 0

    #enumarate in dataloader (btch-size= 6)
    for batch_idx, (s1, s2, y) in enumerate(dataLoader):
        #Data to GPU
        s1 = s1.to(device)
        s2 = s2.to(device)
        y = y.to(device)

        #set up optimizer
        optimizer.zero_grad()

        #feed images to net
        s1, s2 = model(s1, s2)

        #calculate loss
        result = loss(s1, s2, y)
        signs += len(s1)
        calculated_loss += result.item() * len(s1)

        #print loss
        if (batch_idx + 1) % 30 == 0 or batch_idx == len(dataLoader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataLoader), calculated_loss / signs))

        #propagat backward
        result.backward()
        optimizer.step()

#evaluation of the model
@torch.no_grad()
def eval(model, device, dataLoader, loss):
    model.eval()

    # for loss and accuaracy
    distances = []
    calculated_loss = 0
    signs = 0

    #enumerate in dataloader
    for batch_idx, (s1, s2, y) in enumerate(dataLoader):
        #data to GPU
        s1 = s1.to(device)
        s2 = s2.to(device)
        y = y.to(device)

        #feed images to model
        s1, s2 = model(s1, s2)
        result = loss(s1, s2, y)
        distances.extend(zip(torch.pairwise_distance(s1, s2, 2).cpu().tolist(), y.cpu().tolist()))

        signs += len(s1)
        calculated_loss += result.item() * len(s1)

        #print loss
        if (batch_idx + 1) % 30 == 0 or batch_idx == len(dataLoader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataLoader), calculated_loss / signs))

    #calculate accuracy
    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy = accuracy(distances, y)
    print(f'Max  val_accuracy: {max_accuracy}')
    return calculated_loss / signs, max_accuracy
