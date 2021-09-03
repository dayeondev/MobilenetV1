from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def show_images(images):
    # show(torchvision.utils.make_grid(images))
    plt.imshow(torchvision.utils.make_grid(images, normalize=True).permute(1, 2, 0).cpu())
    plt.show()