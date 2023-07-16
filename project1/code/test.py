import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from main import CelebAMaskDataset
from mmseg.apis import init_segmentor, inference_segmentor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--out', default=None, type=str)
    args = parser.parse_args()
    print(args)

    model = init_segmentor(args.cfg, args.ckpt, device='cuda:0')
    model.CLASSES = CelebAMaskDataset.CLASSES
    model.PALETTE = CelebAMaskDataset.PALETTE
    img_dir = '../data/test/test_image/'
    out_dir = '../submit/' + args.out + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(1000)):
        img = img_dir + str(i) + '.jpg'
        result = inference_segmentor(model, img)
        im = Image.fromarray(np.uint8(np.squeeze(np.array(result))))
        im.save(out_dir + str(i) + '.png')


