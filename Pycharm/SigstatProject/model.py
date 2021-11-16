import torch
from torch import nn
import img_read as read
from torchvision import transforms
from PIL import ImageOps


class SigSiameseNet(torch.nn.Module):
    '''
    stride= eltolas merteke
    dialation= mennyi közt hagyjon

    kernel_size= hagy kockabol lesz egy uj kocka
    kimenet_mérete= bemenet-kernel_size+1

    kernelek_száma= 96, ennyi darab uj "kép" fog összejönni

    padding= körül lehet venni  a képet egy csupa 0 paddingel egy a konvolúció után +1 soros es oszlopos lesz
    0000
    0230
    0510
    0000

    '''

    # input image [462x1133]

    def __init__(self):
        super().__init__()

        self.cnnModel = nn.Sequential(
            # input shape megallapitasa (img_w, img_h,1)
            nn.Conv2d(1, 96, kernel_size=11),
            # (kimenet_mérete= bemenet-kernel_size+1, mindket dimenzioban) [452,1123]
            nn.ReLU(),  # relu aktivációs fv
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            # batch_normaliazation padding=(2,2)lehet meg kene ide
            nn.MaxPool2d(3, stride=2),  # valszeg a fele lesz a kimenet meg nem tudom. [226,561]

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),
            # (marad +4 a padding miatt, -5 kernel size +1 )
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(3, stride=2),  # [113,280]
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            # (marad +2 a padding miatt, -3 kernel size +1 )
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # [56,140]
            nn.Dropout2d(p=0.3),

            nn.Flatten(),
            nn.Linear(2007040, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU()

        )

    def forward(self, in1):
        in1 = self.cnnModel(in1)
        # in2= cnnModel(in2)
        return in1  # , in2


# print('Using {} device'.format(device))
#
# model = SigSiameseNet().to(device)
# print(model)

# X = torch.rand(452, 1123)

# # Loading signs
#
# DataLoader = read.DataLoader()
#
# DataLoader.load()
# genuen_images = DataLoader.genuine_images
#
# # print(genuen_images[2][2])
#
#
# img = genuen_images[2][2]
# img2 = genuen_images[2][3]
#
# data = []
# x_np = torch.from_numpy(img)
# data.append(x_np)
# x_np2 = torch.from_numpy(img2)
#
# image_transform = transforms.Compose(
#     [transforms.Resize((155, 220)),
#      ImageOps.invert, transforms.ToTensor()]
# )
# current_pic = image_transform(img)
# logits = model(current_pic)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")
