import torch
import model as SiameseModel
import loss as CL
import img_read
import train
import numpy as np
import random


#eszkoz letrehozasa, init
torch.manual_seed(444)
torch.cuda.manual_seed_all(444)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
#model attoltese
model = SiameseModel.SigSiameseNet().to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

epoch_num = 20

#ez az altalunk irt dataloader, nem beepitettet hasznalok
DataLoader = img_read.DataLoader()

DataLoader.load()

#Összes valos eredeti es hamis alairas
genuine_images = DataLoader.genuine_images
forged_images = DataLoader.forged_images

# TODO test train split
train_pairs, test_pairs = img_read.createPairs(genuine_images, forged_images)
model.train()
#print(model)

# teszt tomb megnezni hogy jol mukodik-e a model


# futtatas
for epoch in range(epoch_num):
    print('Epoch {}/{}'.format(epoch, epoch_num))
    print('Training', '-' * 20)
    train.train(model, optimizer, device, train_pairs)
    print('Evaluating', '-' * 20)
    train.eval(model, device, test_pairs)
    scheduler.step()


    to_save = {
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optim': optimizer.state_dict(),
    }

    print('Saving checkpoint..')
   # torch.save(to_save, 'checkpoints/epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, loss, acc))

print('Done')
