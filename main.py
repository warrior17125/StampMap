# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset
# from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Training schedule
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train_batch_size', default=2, type=int)
    parser.add_argument('--infer_batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--warmup_type', default="linear", type=str)
    parser.add_argument('--warmup_iters', default=2000, type=int)
    parser.add_argument('--random_seed', default=666, type=int)
    parser.add_argument('--clip_gradient', default=5.0, type=float)

    # Dataloader
    parser.add_argument('--num_worker', default=12, type=int)
    parser.add_argument('--worker_prefetch_factor', default=12, type=int)
    parser.add_argument('--shuffle_data', default=True, type=bool)
    
    # Stamp Map
    parser.add_argument('--sample_history_frame_size', default=15, type=int)
    parser.add_argument('--sub_clip_sample_size', default=5, type=int)
    parser.add_argument('--sample_interval', default=1, type=int)
    parser.add_argument('--road_element_pad_num', default=50, type=int)
    parser.add_argument('--road_pt_pad_num', default=350, type=int)

    parser.add_argument('--pad_pt_value_det', default=0.0, type=float)
    parser.add_argument('--pad_pt_value_gt', default=-1.0, type=float)
    parser.add_argument('--pad_attr_value', default=0.0, type=float)
    parser.add_argument('--road_element_cls', default={"background": 0, "curb": 1, "lane": 2, "stopline": 3}, type=dict)
    parser.add_argument('--lane_type', default={
        "solid": 0,
        "dash": 1,
        "solid_solid": 2,
        "solid_dash": 3,
        "dash_solid": 4,
        "dash_dash": 5,
        "deceleration_solid": 6,
        "deceleration_dash": 7,
        "guide": 8,
        "curb": 9,
        "other": 10,
        "ignore": -1,
    }, type=dict)
    parser.add_argument('--lane_color', default={
        "white": 0,
        "yellow": 1,
        "white_white": 2,
        "white_yellow": 3,
        "yellow_white": 4,
        "yellow_yellow": 5,
        "orange": 6,
        "blue": 7,
        "other": 8,
        "ignore": -1,
    }, type=dict)
    parser.add_argument('--curb_type', default={"cross": 0, "uncross": 1, "ignore": -1}, type=dict)
    parser.add_argument('--curb_subtype', default={
        "ignore": -1,
        "cement_block": 0,
        "general_fence": 1,
        "cone": 2,
        "barrel": 3,
        "barrier": 4,
        "grass": 5,
        "gravel": 6,
        "wall_flat": 7,
        "other": 8,
    }, type=dict)
    parser.add_argument('--road_element_attr_dim', default=5, type=int)
    parser.add_argument('--road_element_attr_def', default={
        "curb_type": 0,
        "curb_subtype": 1,
        "lane_type": 2,
        "lane_color": 3,
        "delta_ts": 4,
    }, type=dict)
    parser.add_argument("--bev_range", default=[0.0, 200.0, -50.0, 50.0], type=list)
    parser.add_argument('--attrs_range', default={
        "class": [0, 3],
        "lane_type": [0, 10],
        "lane_color": [0, 8],
        "curb_type": [0, 1],
        "curb_subtype": [0, 8],
    }, type=dict)

    # Transformer
    parser.add_argument("--dim_model", default=256, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--encoder_num_layers", default=6, type=int)
    parser.add_argument("--decoder_num_layers", default=6, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--num_track_query", default=50, type=int)

    # Match weights
    parser.add_argument('--match_weights', default={"class_weight": 1.0, "pt_confidence_weight": 1.0, "pt_coord_weight": 2.0,}, type=dict)
    
    # Loss weights
    parser.add_argument('--loss_weights', default={"class_weight": 3.0, "pt_confidence_weight": 1.0, "pt_coord_weight": 3.0,}, type=dict)
    parser.add_argument('--label_loss_weight_background', default=0.25, type=float)  # alpha in Focal Loss
    parser.add_argument('--point_confidence_mask_threshold', default=0.5, type=float)

    # Track query
    parser.add_argument('--query_score_threshold', default=0.6, type=float)
    parser.add_argument('--query_filter_score_threshold', default=0.4, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)


    # Others
    parser.add_argument('--data_path', default='datasets/Lane')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')

    return parser


def main(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('STAMP_MAP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
