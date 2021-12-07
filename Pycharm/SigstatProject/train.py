import torch


# train fuggveny batchekkel meg nem foglalkozik, sem a hiba kiertekelesevel, eloszor csak atakartam rajta futtatni par adatot

def train(model, optimizer, device, data, loss):
    # TODO nem ertem hogy mukodik a to device es nem tudom, hogy ha egy masik file-ben atrakom a valtozot
    # es atadom parameterkent megtartja-e a tulajdonsagat

    print(loss)
    calculated_loss = 0
    samples = 0

    model.train()
    # TODO batch implementalasa

    # az osszes adatot atkuljuk a halon, (kep1, kep1,y)
    for pair in data:
        # TODO valoszinuleg ez nem mukodik itt jol
        s1 = torch.tensor(pair[0]).float().to(device)
        s2 = torch.tensor(pair[1]).float().to(device)

        # Itt az y erteke teljesen megvaltozik valmire, nem tudom hogy a to(Device) vagy a ShortTensor rontja el
        y = torch.tensor(pair[2]).to(device)
        print(y)
        print(pair[2])
        print(s1.shape)

        optimizer.zero_grad()
        s1, s2 = model(s1, s2)
        result = loss(s1, s2, y)  # itt dobja a hibat
        # Boolean value of Tensor with more than one value is ambiguous

        result.backward()
        optimizer.step()


# eval fv, idaig el se jutunk
@torch.no_grad()
def eval(model, device, data, loss):
    model.eval()

    for pair in data:
        s1 = torch.tensor(pair[0]).float().to(device)
        s2 = torch.tensor(pair[1]).float().to(device)
        y = torch.tensor(pair[2]).to(device)

        s1, s2 = model(s1, s2)
        result = loss(s1, s2, y)

    # TODO calculate accuracy
    return
