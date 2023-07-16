import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmedit.apis import single_gpu_test
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.cfg)
    cfg.gpus = 1
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.ckpt, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, save_image=True,
                          save_path='../result/' + args.cfg + '/HQ/')
