import glob
import numpy as np
import PIL
import random
import csv


class PrepSiameseData:

    def __init__(self):
        self.genuine_images = []
        self.forged_images = []

    def load(self):
        # loading 5 test images
        for folder_path in glob.glob('data/Train/*'):
            signer_genuine = []
            signer_forged = []
            for img_path in glob.glob(folder_path + '/*.png'):
                # beolvasas atmeretezes atalakitas
                # 150x250
                sign = PIL.Image.open(img_path).convert('L')
                # print(sign.size)
                resized_sign = sign.resize((75, 125))
                # print(resized_sign)
                resized_sign = np.array(resized_sign)
                resized_sign = resized_sign[np.newaxis, ...]

                # trans_sign = np.transpose(resized_sign, (2, 0, 1))

                if img_path.find("Genuine") != -1:
                    signer_genuine.append(resized_sign)
                if img_path.find("Forged") != -1:
                    signer_forged.append(resized_sign)
            self.genuine_images.append(signer_genuine)
            self.forged_images.append(signer_forged)


# eredeti-eredet --> 1, eredeti-hamis --> 2 párok kialakítása
def createPairs(genuine_images, forged_images):
    train_genuine = []
    test_genuine = []
    train_forged = []
    test_forged = []

    for signer in range(len(genuine_images)):
        signers_sign = []
        for i in range(len(genuine_images[signer]) - 1):
            for j in range(i + 1, len(genuine_images[signer])):
                signers_sign.append([genuine_images[signer][i], genuine_images[signer][j], 1])
        train_genuine, test_genuine = Split(signers_sign)
    # egy alairohoz 190 ilyen genuine par keletkezik
    # osszesen 20 * 190 = 3800

    for signer in range(len(genuine_images)):
        signers_sign = []
        for i in range(len(genuine_images[signer])):
            for j in range(len(forged_images[signer])):
                signers_sign.append([genuine_images[signer][i], genuine_images[signer][j], 0])
        train_forged, test_forged = Split(signers_sign)

    # egy alairohoz 400 ilyen forged par keletkezik
    # osszesen 20* 400 = 8000

    # 11800 képunk lesz

    train_data = []
    train_data.extend(train_genuine)
    train_data.extend(train_forged)
    random.shuffle(train_data)
    test_data = []
    test_data.extend(test_genuine)
    test_data.extend(test_forged)
    random.shuffle(test_data)

    return train_data, test_data


def Split(array_to_split, size=0.8):
    train_result = []
    test_result = []
    random.shuffle(array_to_split)
    split = int(len(array_to_split) * size)
    train_result.extend(array_to_split[:split])
    test_result.extend(array_to_split[split:])
    return train_result, test_result
