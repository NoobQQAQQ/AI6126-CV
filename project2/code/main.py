import os.path as osp
import argparse

from mmcv import Config, mkdir_or_exist
from mmcv.runner import set_random_seed
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str)
    args = parser.parse_args()
    print(args)

    # read a config file
    cfg = Config.fromfile(args.cfg)
    # set seed for reproducibility
    set_random_seed(cfg.seed, deterministic=False)
    # set work _dir
    cfg.work_dir = "../result/" + args.cfg
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]
    # Build the restorer
    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # check the number of total parameters
    nr_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(nr_para)
    assert (nr_para <= 1821085)
    # train the restorer
    cfg.gpus = 1
    train_model(model, datasets, cfg, distributed=False, validate=True, meta=dict())
