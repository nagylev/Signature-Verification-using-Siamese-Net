import torch
import model as SiameseModel
import loss_accuracy as CL
import img_read
import train
import create_data as createData

# creating the device
torch.manual_seed(444)
torch.cuda.manual_seed_all(444)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# checking if we have cuda
print(torch.cuda.is_available())
# create model and send to device
model = SiameseModel.SigSiameseNet().to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

loss = CL.ContrastiveLoss(1, 1, 1).to(device)
epoch_num = 20

# loading the images
DataLoader = img_read.PrepSiameseData()
DataLoader.load()

# get the forged and genuine images
genuine_images = DataLoader.genuine_images
forged_images = DataLoader.forged_images

# separating test and train data
train_pairs, test_pairs = img_read.createPairs(genuine_images, forged_images)
train_load = createData.data_loader(train_pairs, 6)
test_load = createData.data_loader(test_pairs, 6)

# start train
model.train()

# teszt tomb megnezni hogy jol mukodik-e a model

# running 20 epoch training
for epoch in range(epoch_num):
    print('Epoch {}/{}'.format(epoch, epoch_num))
    print('Training', '-' * 40)
    train.train(model, optimizer, device, train_load, loss)
    print('Evaluating', '-' * 40)
    lost, acc = train.eval(model, device, test_load, loss)
    scheduler.step()

print('Done')
