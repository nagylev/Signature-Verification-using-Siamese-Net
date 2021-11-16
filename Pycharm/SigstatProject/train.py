def train(model, optimizer, loss, device, data):
    model.train()
    # batch implementalasa

    for pair in data[0]:
        print(pair)
        s1 = pair[0].to(device)
        s2 = pair[1].to(device)
        y = pair[2].to(device)

        # optimizeer.zero grad
        s1, s2 = model(s1, s2)
        loss = loss(s1, s2, y)
        loss.backward()
        optimizer.step()





def eval(model, loss, data):
    return
