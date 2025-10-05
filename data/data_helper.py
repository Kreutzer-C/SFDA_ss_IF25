from os.path import join, dirname
import torch
from torch.utils.data import Sampler, DataLoader
from torchvision import transforms
from data.JigsawLoader import get_split_dataset_info, _dataset_info, _dataset_info_pda, _dataset_info_oda
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawIADataset, JigsawTestIADataset_idx, JigsawTestIADataset


def get_train_dataloader(args):
    dataset_list = [args.source]
    assert isinstance(dataset_list, list)
    train_datasets = []
    val_datasets = []
    for dname in dataset_list:
        if args.dataset == 'PACS':
            name_train, labels_train = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_train_kfold.txt' % dname))
            name_val, labels_val = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_crossval_kfold.txt' % dname))
        else:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(
                join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_train.txt' % dname), args.val_size)
        train_dataset = JigsawIADataset(name_train, labels_train, args.data_path, img_transformer=get_train_transformers(args))
        train_datasets.append(train_dataset)

        val_dataset = JigsawTestIADataset(name_val, labels_val, args.data_path, img_transformer=get_val_transformer(args))
        val_datasets.append(val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    # batch_sampler = ClassBalancedBatchSampler(train_dataset, batch_size=args.batch_size)
    # train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def get_ODA_train_dataloader(args):
    dataset_list = [args.source]
    assert isinstance(dataset_list, list)
    train_datasets = []
    val_datasets = []
    for dname in dataset_list:
        if args.dataset == 'PACS':
            name_train, labels_train = _dataset_info_oda(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_train_kfold.txt' % dname), args.src_classes)
            name_val, labels_val = _dataset_info_oda(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_crossval_kfold.txt' % dname), args.src_classes)
        else:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(
                join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_train.txt' % dname), args.val_size, oda=True, src_classes=args.src_classes)
        train_dataset = JigsawIADataset(name_train, labels_train, args.data_path, img_transformer=get_train_transformers(args))
        train_datasets.append(train_dataset)

        val_dataset = JigsawTestIADataset(name_val, labels_val, args.data_path, img_transformer=get_val_transformer(args))
        val_datasets.append(val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def get_adapt_dataloader(args):
    if args.dataset == 'PACS':
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test_kfold.txt' % args.target))
    else:
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test.txt' % args.target))
    # names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestIADataset(names, labels, args.data_path, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader

def get_PDA_adapt_dataloader(args):
    if args.dataset == 'PACS':
        names, labels = _dataset_info_pda(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test_kfold.txt' % args.target), args.tgt_classes)
    else:
        names, labels = _dataset_info_pda(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test.txt' % args.target), args.tgt_classes)

    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestIADataset(names, labels, args.data_path, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader


def get_adapt_dataloader_idx(args):
    if args.dataset == 'PACS':
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test_kfold.txt' % args.target))
    else:
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test.txt' % args.target))
    # names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestIADataset_idx(names, labels, args.data_path, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader


def get_test_dataloader(args):
    if args.dataset == 'PACS':
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test_kfold.txt' % args.target))
    else:
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test.txt' % args.target))
    # names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestIADataset(names, labels, args.data_path, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return loader

def get_PDA_test_dataloader(args):
    if args.dataset == 'PACS':
        names, labels = _dataset_info_pda(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test_kfold.txt' % args.target), args.tgt_classes)
    else:
        names, labels = _dataset_info_pda(join(dirname(__file__), 'data_path_txt_lists', args.dataset, '%s_test.txt' % args.target), args.tgt_classes)

    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestIADataset(names, labels, args.data_path, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(img_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

