import os.path as osp
import argparse

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config, mkdir_or_exist
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


@DATASETS.register_module()
class CelebAMaskDataset(CustomDataset):
    CLASSES = ('background', 'skin', 'nose', 'eye_g', 
                'l_eye', 'r_eye', 'l_brow', 'r_brow', 
                'l_ear', 'r_ear', 'mouth', 'u_lip', 
                'l_lip', 'hair', 'hat', 'ear_r', 
                'neck_l', 'neck', 'cloth')
    PALETTE = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], 
                [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], 
                [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], 
                [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str)
    args = parser.parse_args()
    print(args)

    # read a config file
    cfg = Config.fromfile(args.cfg + '.py')
    # set seed for reproducibility
    set_random_seed(cfg.seed, deterministic=False)
    # set work _dir
    cfg.work_dir = "../result/" + args.cfg
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]
    # Build the segmentor
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Add attributes for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE
    # Create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
    # Inference with trained model
    # img = mmcv.imread('iccv09Data/images/6000124.jpg')
    # model.cfg = cfg
    # result = inference_segmentor(model, img)
    # plt.figure(figsize=(8, 6))
    # show_result_pyplot(model, img, result, palette)
