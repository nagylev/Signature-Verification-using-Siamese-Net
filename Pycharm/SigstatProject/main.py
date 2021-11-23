import torch
import model as SiameseModel
import loss as CL
import img_read as read
import train
import numpy as np

torch.manual_seed(444)
torch.cuda.manual_seed_all(444)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SiameseModel.SigSiameseNet().to(device)


# scheduler??
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

epoch_num = 20

DataLoader = read.DataLoader()

DataLoader.load()
genine_images = DataLoader.genuine_images
forged_images = DataLoader.forged_images

# TODO test train split
genuine_pairs, forged_pairs = read.createPairs(genine_images, forged_images)
model.train()
print(model)

for epoch in range(epoch_num):
    print('Epoch {}/{}'.format(epoch, epoch_num))
    print('Training', '-' * 20)
    train.train(model, optimizer, device, genuine_pairs) #osszdata train
    print('Evaluating', '-' * 20)
    # loss, acc = train.eval(model, loss, genuine_pairs) #katasztrofa osszdata teszt
    scheduler.step()

    to_save = {
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optim': optimizer.state_dict(),
    }

    print('Saving checkpoint..')
    torch.save(to_save, 'checkpoints/epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, loss, acc))

print('Done')
