import glob
import numpy as np
import PIL
import random


class PrepSiameseData:

    def __init__(self):
        self.genuine_images = []
        self.forged_images = []

    def load(self):
        signers = 0
        # loading images from data folder
        for folder_path in glob.glob('data/*'):
            signer_genuine = []
            signer_forged = []
            signers += 1
            for img_path in glob.glob(folder_path + '/*.png'):
                # reading each picture and converting them to single channel image
                # 75x125 size of picture after resize
                sign = PIL.Image.open(img_path).convert('L')
                resized_sign = sign.resize((75, 125))
                resized_sign = np.array(resized_sign)
                resized_sign = resized_sign[np.newaxis, ...]

                # add the picture to the right array
                if img_path.find("Genuine") != -1:
                    signer_genuine.append(resized_sign)
                if img_path.find("Forged") != -1:
                    signer_forged.append(resized_sign)
            self.genuine_images.append(signer_genuine)
            self.forged_images.append(signer_forged)

        print("There are {} signers, with {} genuine and {} forged signs each".format(signers,
                                                                                      len(self.genuine_images[0]),
                                                                                      len(self.forged_images[0])))


# creating pairs
# 1, genuine,genuine
# 0, genuine,forged
def createPairs(genuine_images, forged_images):
    train_genuine = []
    test_genuine = []
    train_forged = []
    test_forged = []
    signers_sign_count = 0
    signer_count = 0

    for signer in range(len(genuine_images)):
        signers_sign = []
        signer_count += 1
        for i in range(len(genuine_images[signer]) - 1):
            for j in range(i + 1, len(genuine_images[signer])):
                signers_sign.append([genuine_images[signer][i], genuine_images[signer][j], 1])

        # splitting test and train sets
        if signer_count < 32:
            train_genuine.extend(signers_sign)
        if signer_count >= 32:
            test_genuine.extend(signers_sign)
        signers_sign_count = len(signers_sign)
    # one signer should have 190 genuine genuine pair
    # total of 20  190 = 3800
    print('After creating pairs, one signer {} genuine-genuine pairs each, for all signers {}'.format(
        signers_sign_count,
        (len(
            train_genuine) + len(
            test_genuine))))

    signer_count = 0
    for signer in range(len(genuine_images)):
        signers_sign = []
        signer_count += 1
        for i in range(len(genuine_images[signer])):
            for j in range(len(forged_images[signer])):
                signers_sign.append([genuine_images[signer][i], genuine_images[signer][j], 0])
        if signer_count < 2:
            train_forged.extend(signers_sign)
        if signer_count >= 2:
            test_forged.extend(signers_sign)
        signers_sign_count = len(signers_sign)

    # one signer should have 400 genuine forged pair
    # total of 20 * 400 = 8000
    print('After creating pairs, one signer {} genuine-forged pairs each, for all signers {}'.format(
        signers_sign_count, (len(
            train_forged) + len(test_forged))))

    # we`ll have 11800 pictures in total

    # compile train and test data
    train_data = []
    train_data.extend(train_genuine)
    train_data.extend(train_forged)
    random.shuffle(train_data)
    test_data = []
    test_data.extend(test_genuine)
    test_data.extend(test_forged)
    random.shuffle(test_data)

    return train_data, test_data

# split the array in the given ratio
def Split(array_to_split, size=0.8):
    train_result = []
    test_result = []
    random.shuffle(array_to_split)
    split = int(len(array_to_split) * size)
    train_result.extend(array_to_split[:split])
    test_result.extend(array_to_split[split:])
    return train_result, test_result
