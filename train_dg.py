import logging
import os
import pprint
from tkinter import image_names
import torch.nn.functional as F
import torch
import yaml
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import time
from dataset.ssdg_dataset import SSDGDataset
from evaluate import evaluate
from model import factory
from utils import AverageMeter, DiceLoss, fix_seed, init_log
from utils.env import get_module_version
from utils.mask_convert import converter
from utils.parse_args import parse_args
from utils.sampler import MultiDomainSampler
from PIL import Image 
import numpy as np

def main():
    args = parse_args()

    fix_seed(args.seed)

    cfg: dict = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg.update(yaml.load(open(args.shared_config, "r"), Loader=yaml.Loader))
    cfg.update(yaml.load(open(args.train_config, "r"), Loader=yaml.Loader))

    torch.set_num_threads(cfg["num_threads"])

    convert = converter[cfg["dataset"]]
    trainset = SSDGDataset(name=cfg["dataset"],
                           root=cfg["data_root"],
                           target_domain=args.domain,
                           mode="train",
                           n_domains=cfg["n_domains"],
                           image_size=cfg["image_size"])
    _, trainset_l, indices = trainset.split_ulb_lb(args.ratio)

    valset = SSDGDataset(name=cfg["dataset"],
                         root=cfg["data_root"],
                         target_domain=args.domain,
                         mode="val",
                         n_domains=cfg["n_domains"],
                         image_size=cfg["image_size"])
    # for labeled data evaluation
    valset_l = trainset_l.validation()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0  # type: ignore

    logger.info("labeled: \n{}".format(trainset_l))

    env = get_module_version([
        "numpy",
        "PIL",
        "scipy",
        "skimage",
        "torch",
        "torchvision",
    ])
    env = {"env": env}
    all_args = {**cfg, **vars(args), **env}
    logger.info("cfg: \n{}\n".format(pprint.pformat(all_args)))

    os.makedirs(args.save_path, exist_ok=True)

    with open(args.save_path + "/split", "w") as f:
        f.write(str(indices))

    model = factory(args.model, cfg["n_channels"], cfg["n_classes"])
    model_enhance = factory("VMUnet_enhance", cfg["n_channels"], cfg["n_classes"])
    model.load_from()
    model_enhance.load_from()
    model.cuda()
    model_enhance.cuda()

    if cfg["optimizer"] == "sgd":
        optimizer = SGD(model.parameters(), cfg["lr"], momentum=0.9)
    elif cfg["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), cfg["lr"])
        optimizer_enhance = AdamW(model_enhance.parameters(), cfg["lr"])
    elif cfg["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), cfg["lr"])
    else:
        raise NotImplementedError

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["n_classes"])
    criterion_consistency = nn.MSELoss()
    dice_args = dict(softmax="softmax", onehot=True)

    sampler = MultiDomainSampler(trainset_l.lengths, balanced=cfg["balanced"])
    trainloader = DataLoader(trainset_l,
                             batch_size=cfg["batch_size"],
                             pin_memory=True,
                             num_workers=cfg["num_workers"],
                             drop_last=True,
                             sampler=sampler)
    trainloader = iter(trainloader)

    valloader = DataLoader(valset,
                           batch_size=1,
                           pin_memory=True,
                           num_workers=1,
                           drop_last=False)
    valloader_l = DataLoader(valset_l,
                             batch_size=1,
                             pin_memory=True,
                             num_workers=1,
                             drop_last=False)

    n_iters = cfg["iters"]
    total_iters = n_iters * cfg["epochs"]
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        checkpoint_enhance = torch.load(os.path.join(args.save_path, "latest_enhance.pth"))
        model.load_state_dict(checkpoint["model"])
        model_enhance.load_state_dict(checkpoint_enhance["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_enhance.load_state_dict(checkpoint_enhance["optimizer"])
        epoch = checkpoint["epoch"]

        """if epoch >= cfg["epochs"] - 1:
            logger.info("************ Skip trained checkpoint at epoch %i\n" %
                        epoch)
            exit()"""

        # reset learning rate
        current_iters = epoch * n_iters
        lr = cfg["lr"] * (1 - current_iters / total_iters)**0.9
        optimizer.param_groups[0]["lr"] = lr
        optimizer_enhance.param_groups[0]["lr"] = lr

        logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    #起始注释处
    writer = SummaryWriter(args.save_path)

    for epoch in range(epoch + 1, cfg["epochs"]):
        model.train()
        total_loss = AverageMeter()
        
        start_train_time = time.time()

        for i in range(n_iters):
            # make sure batch size is the same
            image, _, mask = next(trainloader)
            image, mask = image.cuda(), mask.cuda()
            mask = convert(mask)

            pred = model(image)
            pred_enhance = model_enhance(image)
            loss_ce = (criterion_ce(pred, mask) + criterion_ce(pred_enhance, mask)) / 2.0
            loss_dice = (criterion_dice(pred, mask, **dice_args) + criterion_dice(pred_enhance, mask, **dice_args)) / 2.0
            loss_consistency = criterion_consistency(pred, pred_enhance)
            loss = loss_ce + loss_dice + 0.1 * loss_consistency


            optimizer.zero_grad()
            optimizer_enhance.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_enhance.step()

            total_loss.update(loss.item())

            current_iters = epoch * n_iters + i
            lr = cfg["lr"] * (1 - current_iters / total_iters)**0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer_enhance.param_groups[0]["lr"] = lr

            if i % (n_iters // 8) == 0:
                logger.info("Iters: {:}, Total loss: {:.3f}".format(
                    i, total_loss.avg))
                writer.add_scalar("train/loss_all", total_loss.avg,
                                  current_iters)
        
        # 记录训练结束时间并计算耗时
        end_train_time = time.time()
        train_duration = end_train_time - start_train_time
        logger.info("Training time for epoch {}: {:.2f} seconds".format(epoch, train_duration))

        # 进行评估，并记录评估时间
        start_eval_time = time.time()
        
        # target domain
        mean_dice, _, dice_class_domain = evaluate(model,
                                                   valloader,
                                                   cfg,
                                                   is_target_domain=True)
        
        end_eval_time = time.time()
        eval_duration = end_eval_time - start_eval_time
        logger.info("Evaluation time for epoch {}: {:.2f} seconds".format(epoch, eval_duration))

        dice_class = dice_class_domain[0]

        # target domain enhance
        mean_dice_enhance, _, dice_class_domain_enhance = evaluate(model_enhance,
                                                   valloader,
                                                   cfg,
                                                   is_target_domain=True)
        dice_class_enhance = dice_class_domain_enhance[0]

        # labeled source domains
        mean_dice_l, mean_dice_l_domain, _ = evaluate(model, valloader_l, cfg)
        mean_dice_l_enhance, mean_dice_l_domain_enhance, _ = evaluate(model_enhance, valloader_l, cfg)
        for _domain in range(cfg["n_domains"] - 1):
            writer.add_scalar("eval_l/Domain_%d_dice" % _domain,
                              mean_dice_l_domain[_domain], epoch)
            writer.add_scalar("eval_l_enhance/Domain_%d_dice" % _domain,
                              mean_dice_l_domain_enhance[_domain], epoch)

        for (cls_idx, dice) in enumerate(dice_class):
            logger.info("***** Evaluation ***** >>>> "
                        "Class [{:}] Dice: {:.2f}".format(cls_idx, dice))
        logger.info("***** Evaluation ***** >>>> "
                    "MeanDice: {:.2f}".format(mean_dice))
        logger.info("***** Evaluation ***** >>>> "
                    "MeanDice_l: {:.2f}\n".format(mean_dice_l))

        for (cls_idx, dice) in enumerate(dice_class_enhance):
            logger.info("***** Evaluation_enhance ***** >>>> "
                        "Class [{:}] Dice: {:.2f}".format(cls_idx, dice))
        logger.info("***** Evaluation_enhance ***** >>>> "
                    "MeanDice: {:.2f}".format(mean_dice_enhance))
        logger.info("***** Evaluation_enhance ***** >>>> "
                    "MeanDice_l: {:.2f}\n".format(mean_dice_l_enhance))

        writer.add_scalar("eval/MeanDice", mean_dice, epoch)
        writer.add_scalar("eval_enhance/MeanDice", mean_dice_enhance, epoch)

        for i, dice in enumerate(dice_class):
            writer.add_scalar("eval/Class_%s_dice" % i, dice, epoch)
        
        for i, dice in enumerate(dice_class_enhance):
            writer.add_scalar("eval_enhance/Class_%s_dice" % i, dice, epoch)
        torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
        torch.save(checkpoint_enhance, os.path.join(args.save_path, "latest_enhance.pth"))
    writer.close()
    #尾巴注释处


if __name__ == "__main__":
    main()
