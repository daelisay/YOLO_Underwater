import os
import time
import torch
import torch.nn as nn

from dataloader.DUODataset import DUODataset, collate_fn
from torch.utils.data import DataLoader
from models.select_model import select_model

from utils.opts import Opt
from utils.logger import Logger
from utils.utils import seed_torch

from train import train
from val import val
from test import test

if __name__ == "__main__":
    opt = Opt().parse()
    opt.device = torch.device("cuda" if torch.cuda.is_available() and opt.gpu else "cpu")  # <-- DI SINI

    opt.num_classes = 5
    
    seed_torch(opt.manual_seed)

    ########################################
    #              Dataset & Loader        #
    ########################################
    if not opt.no_train:
        train_dataset = DUODataset(
            root_dir=opt.dataset_path,
            annotation_file=os.path.join(opt.dataset_path, "train.json"),
            image_folder='image_folder',  # pastikan ini sama persis dengan nama folder di filesystem
            split='train',
            image_size=opt.image_size,
            use_augmentation=True,
            box_type='yolo',
            cache=opt.cache,
            preprocessing=opt.preprocessing)
        train_loader = DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=opt.num_threads, pin_memory=True)
        train_logger = Logger(os.path.join(opt.checkpoint_path, 'train.log'))

    if not opt.no_val:
        val_dataset = DUODataset(
            root_dir=opt.dataset_path,
            annotation_file=os.path.join(opt.dataset_path, "val.json"),
            image_folder='image_folder',
            split='val',
            image_size=opt.image_size,
            use_augmentation=False,
            box_type='yolo',
            cache=opt.cache,
            preprocessing=opt.preprocessing)
        val_loader = DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=opt.num_threads)
        val_logger = Logger(os.path.join(opt.checkpoint_path, 'val.log'))

    ########################################
    #                 Model                #
    ########################################
    best_mAP = 0
    torch.manual_seed(opt.manual_seed)
    model = select_model(opt)

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr,
            momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError("Only Adam and SGD are supported")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.model == checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])

        opt.begin_epoch = checkpoint['epoch']
        model = model.to(opt.device)
        if not opt.no_train and not opt.pretrain:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        best_mAP = checkpoint["best_mAP"]

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(opt.device)

    ########################################
    #           Train, Val, Test           #
    ########################################
    if opt.test:
        print("\n---- Testing Model ----")
        test_dataset = DUODataset(
            root_dir=opt.dataset_path,
            annotation_file=os.path.join(opt.dataset_path, "test.json"),
            image_folder='image_folder',
            split='test',
            image_size=opt.image_size,
            use_augmentation=False,
            box_type='yolo',
            preprocessing=opt.preprocessing)
        test_loader = DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=opt.num_threads)

        test(model, test_loader, opt.begin_epoch, opt)
    else:
        for epoch in range(opt.begin_epoch, opt.num_epochs):
            if not opt.no_train:
                print("\n---- Training Model ----")
                train(model, optimizer, scheduler, train_loader, epoch, opt, train_logger, best_mAP=best_mAP)
            if not opt.no_val and (epoch + 1) % opt.val_interval == 0:
                print("\n---- Evaluating Model ----")
                best_mAP = val(model, optimizer, scheduler, val_loader, epoch, opt, val_logger, best_mAP=best_mAP)