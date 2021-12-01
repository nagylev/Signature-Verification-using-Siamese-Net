import glob
import numpy as np
import PIL


class DataLoader:

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
                #150x250
                sign = PIL.Image.open(img_path)
                resized_sign = sign.resize((150, 250))
                resized_sign = np.array(resized_sign)
                resized_sign = resized_sign[np.newaxis, ...]

                trans_sign = np.transpose(resized_sign, (3, 0, 1, 2))

                if img_path.find("Genuine") != -1:
                    signer_genuine.append(trans_sign)
                if img_path.find("Forged") != -1:
                    signer_forged.append(trans_sign)
            self.genuine_images.append(signer_genuine)
            self.forged_images.append(signer_forged)


# eredeti-eredet --> 1, eredeti-hamis --> 2 párok kialakítása
def createPairs(genuine_images, forged_images):
    genuine_pairs = []
    forged_pairs = []

    for signer in range(len(genuine_images)):
        for i in range(len(genuine_images[signer]) - 1):
            for j in range(i + 1, len(genuine_images[signer])):
                genuine_pairs.append([genuine_images[signer][i], genuine_images[signer][j], 1])
    # egy alairohoz 190 ilyen genuine par keletkezik
    # osszesen 20 * 190 = 3800
    for signer in range(len(genuine_images)):
        for i in range(len(genuine_images[signer])):
            for j in range(len(forged_images[signer])):
                forged_pairs.append([genuine_images[signer][i], genuine_images[signer][j], 0])

    # egy alairohoz 400 ilyen forged par keletkezik
    # osszesen 20* 400 = 8000
    return genuine_pairs, forged_pairs