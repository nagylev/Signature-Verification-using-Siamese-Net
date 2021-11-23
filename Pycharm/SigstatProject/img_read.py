import matplotlib.pyplot as plt
import glob
import numpy as np
import csv
import matplotlib.image as mpimg
import PIL

# image scanner


class DataLoader():
    def __init__(self):
        self.genuine_images = []
        self.forged_images = []

    def load(self):
        # loading 5 test images
        for folder_path in glob.glob('data/Train/*'):
            counter = 0
            signer_genuine = []
            signer_forged = []
            for img_path in glob.glob(folder_path + '/*.png'):
               #beolvasas atmeretezes atalakitas
                sign = PIL.Image.open(img_path)
                resized_sign = sign.resize((150, 250))
                resized_sign = np.array(resized_sign)
                resized_sign = resized_sign[np.newaxis,np.newaxis, ...]

                trans_sign = np.transpose(resized_sign, (4, 0, 1, 2, 3))

                if img_path.find("Genuine") != -1:
                    signer_genuine.append(trans_sign)
                if img_path.find("Forged") != -1:
                    signer_forged.append(trans_sign)
            self.genuine_images.append(signer_genuine)
            self.forged_images.append(signer_forged)

    def getGenuine(self):
        return self.genuine_images

    def getForged(self):
        return self.forged_images


# loader = DataLoader()
# loader.load()
# genuine_images = loader.getGenuine()
# forged_images = loader.getForged()
#
# len(genuine_images)
#
#
# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
#
#
# img = genuine_images[2][2]
#
# print(type(img))
# gray = rgb2gray(img)
# plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
# plt.show()


def createPairs(genuine_images, forged_images):
    genuine_pairs = []
    forged_pairs = []

    for signer in range(len(genuine_images)):
        for i in range(len(genuine_images[signer]) - 1):
            for j in range(i + 1, len(genuine_images[signer])):
                genuine_pairs.append([genuine_images[signer][i], genuine_images[signer][j], 1])
    #egy alairohoz 190 ilyen genuine par keletkezik
    #osszesen 20 * 190 = 3800
    for signer in range(len(genuine_images)):
        for i in range(len(genuine_images[signer])):
            for j in range(len(forged_images[signer])):
                forged_pairs.append([genuine_images[signer][i], genuine_images[signer][j], 0])

    # egy alairohoz 400 ilyen forged par keletkezik
    #osszesen 20* 400 = 8000
    return genuine_pairs, forged_pairs

# genuine_pairs, forged_pairs = createPairs()
#
# filename = "data/genuine_genuine_data.csv"
#
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#
#     csvwriter.writerows(genuine_pairs)
#
# filename = "data/genuine_forged_data.csv"
#
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#
#     csvwriter.writerows(forged_pairs)
