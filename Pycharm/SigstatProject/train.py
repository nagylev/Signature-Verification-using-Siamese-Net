import torch
from loss_accuracy import accuracy


def train(model, optimizer, device, dataLoader, loss):
    model.train()
    calculated_loss = 0
    signs = 0

    # az osszes adatot atkuljuk a halon, (kep1, kep1,y)
    for batch_idx, (s1, s2, y) in enumerate(dataLoader):
        s1 = s1.to(device)
        s2 = s2.to(device)

        # Itt az y erteke teljesen megvaltozik valmire, nem tudom hogy a to(Device) vagy a ShortTensor rontja el
        y = y.to(device)

        optimizer.zero_grad()
        s1, s2 = model(s1, s2)
        result = loss(s1, s2, y)  # itt dobja a hibat
        signs += len(s1)
        calculated_loss += result.item() * len(s1)

        if (batch_idx + 1) % 30 == 0 or batch_idx == len(dataLoader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataLoader), calculated_loss / signs))

        result.backward()
        optimizer.step()


@torch.no_grad()
def eval(model, device, dataLoader, loss):
    model.eval()

    distances = []
    calculated_loss = 0
    signs = 0

    for batch_idx, (s1, s2, y) in enumerate(dataLoader):
        s1 = s1.to(device)
        s2 = s2.to(device)
        y = y.to(device)

        s1, s2 = model(s1, s2)
        result = loss(s1, s2, y)
        distances.extend(zip(torch.pairwise_distance(s1, s2, 2).cpu().tolist(), y.cpu().tolist()))

        signs += len(s1)
        calculated_loss += result.item() * len(s1)

        if (batch_idx + 1) % 30 == 0 or batch_idx == len(dataLoader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataLoader), calculated_loss / signs))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy = accuracy(distances, y)
    print(f'Max  val_accuracy: {max_accuracy}')
    return calculated_loss / signs, max_accuracy
