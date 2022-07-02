import logging
import os
import random
import shutil
import collections
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import tqdm
from configs import configuration
from numpy import linalg as LA

import datasets
from collections import OrderedDict
from sklearn import metrics

from utils import stage2_episode, Classifier, open_loss


def main():
    global args
    name = ''
    args = configuration.parser_args()
    filepath = '{}'.format(args.save_path + '{}shot{}'.format(args.meta_test_shot, name))
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    # create model
    if args.arch == 'ConvNet':
        from model.networks.convnet import ConvNet
        encoder = ConvNet()
    elif args.arch == 'Res12':
        from model.networks.res12 import ResNet
        encoder = ResNet()
    elif args.arch == 'Res18':
        from model.networks.res18 import ResNet
        encoder = ResNet()
    else:
        raise ValueError('')

    # load saved model and creat dataset
    load_checkpoint(encoder, args.model_path)
    encoder.cuda(args.gpu)
    train_mean, train_weight, train_sample, val_sample, test_sample = save_stage1(encoder)
    classifier = Classifier(args, train_weight)
    classifier = classifier.cuda(args.gpu)


    # you can just eval the model
    if args.evaluate:
        checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path + '{}shot{}'.format(args.meta_test_shot, name)))
        classifier.load_state_dict(checkpoint['state_dict'])
        acc_m, acc_pm, auroc_m, auroc_pm = open_eval(classifier, test_sample)
        print(f"acc:{acc_m} +- {acc_pm}")
        print(f"auroc:{auroc_m} +- {auroc_pm}")
        return

    criterion = open_loss(0.5, args.meta_train_way, args.meta_train_query, args.train_oway, args.train_oquery, args.gpu)
    optimizer = get_optimizer(classifier)
    cudnn.benchmark = True
    scheduler = get_scheduler(args.epochs, optimizer)
    tqdm_loop = warp_tqdm(list(range(0, args.epochs)))
    best_auroc = -1
    best_epoch = -1
    for epoch in tqdm_loop:
        # creat episode
        train_x, train_y, rest_class = stage2_episode(train_sample, args.meta_train_way, args.meta_train_shot,
                                                      args.meta_train_query, args.do_open,
                                                      args.train_oway, args.train_oshot, args.train_oquery)
        train_x = torch.from_numpy(train_x).cuda(args.gpu)
        train_y = torch.from_numpy(np.array(train_y)).cuda(args.gpu)
        rest_class = torch.from_numpy(np.array(rest_class)).cuda(args.gpu)

        support_x = train_x[:args.meta_train_way * args.meta_train_shot]
        support_y = train_y[:args.meta_train_way * args.meta_train_shot]
        query_x = train_x[args.meta_train_way * args.meta_train_shot:]
        query_y = train_y[args.meta_train_way * args.meta_train_shot:]
        for j in range(args.meta_train_way):
            support_y[j * args.meta_train_shot:(j + 1) * args.meta_train_shot] = j
            query_y[j * args.meta_train_query:(j + 1) * args.meta_train_query] = j
        query_y[args.meta_train_way * args.meta_train_query:] = -1

        one_hot_labels = torch.zeros(args.meta_train_way * args.meta_train_shot, args.meta_train_way).cuda(
            device=args.gpu).scatter_(1, support_y.view(-1, 1), 1)

        support_x = support_x.view(1, support_x.size(0), support_x.size(1))
        rest_class = rest_class.view(1, rest_class.size(0))
        query_x = query_x.view(1, query_x.size(0), query_x.size(1))
        one_hot_labels = one_hot_labels.view(1, one_hot_labels.size(0), one_hot_labels.size(1))

        # forward to get the result
        cls_scores= classifier(
            features_test=query_x,
            Kbase_ids=rest_class,
            features_train=support_x,
            labels_train=one_hot_labels)

        # eval
        is_best = False
        if (epoch + 1) % args.meta_val_interval == 0:  # or epoch == 0
            acc_m, acc_pm, auroc_m, auroc_pm= open_eval(classifier, val_sample)
            print(f"acc:{acc_m} +- {acc_pm}")
            print(f"auroc:{auroc_m} +- {auroc_pm}")
            if best_auroc < auroc_m:
                best_auroc = auroc_m
                is_best = True
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': classifier.state_dict(),
                    'best_auroc': best_auroc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, epoch + 1, folder=args.save_path + '{}shot{}'.format(args.meta_train_shot,name))
            print(f"best_auroc:{best_auroc} in {best_epoch}")

        # backward
        loss = criterion(cls_scores, query_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def open_eval(classifier, test_sample):
    '''
    eval the model
    inpput:
        classifier:    model
        test_sample:   dataset used to test
    output:
        acc_m:         accuracy
        acc_pm:        95% percent confidence interval for accuracy
        auroc_m:       auroc
        auroc_pm:      95% percent confidence interval for auroc
    '''
    total_correct = 0
    total_num = 0
    acc = []
    all_label = []
    all_dist = []
    correct_dist = []
    unknown_dist = []
    classifier.eval()

    with torch.no_grad():
        for epoch in warp_tqdm(list(range(args.meta_test_iter))):
            # creat episode
            test_x, test_y, _ = stage2_episode(test_sample, args.meta_test_way, args.meta_test_shot,
                                               args.meta_test_query, True,
                                               args.test_oway, args.test_oshot, args.test_oquery)
            base_class = np.array(list(range(args.num_classes)))
            test_x = torch.from_numpy(test_x).cuda(args.gpu)
            test_y = torch.from_numpy(np.array(test_y)).cuda(args.gpu)
            base_class = torch.from_numpy(np.array(base_class)).cuda(args.gpu)

            support_x = test_x[:args.meta_test_way * args.meta_test_shot]
            support_y = test_y[:args.meta_test_way * args.meta_test_shot]
            query_x = test_x[args.meta_test_way * args.meta_test_shot:]
            query_y = test_y[args.meta_test_way * args.meta_test_shot:]

            for j in range(args.meta_test_way):
                support_y[j * args.meta_test_shot:(j + 1) * args.meta_test_shot] = j
                query_y[j * args.meta_test_query:(j + 1) * args.meta_test_query] = j
            query_y[args.meta_test_way * args.meta_test_query:] = 5

            one_hot_labels = torch.zeros(args.meta_test_way * args.meta_test_shot, args.meta_test_way).cuda(
                device=args.gpu).scatter_(1, support_y.view(-1, 1), 1)

            support_x = support_x.view(1, support_x.size(0), support_x.size(1))
            base_class = base_class.view(1, base_class.size(0))
            query_x = query_x.view(1, query_x.size(0), query_x.size(1))
            one_hot_labels = one_hot_labels.view(1, one_hot_labels.size(0), one_hot_labels.size(1))

            # forward to get the result
            cls_scores= classifier(
                features_test=query_x,
                Kbase_ids=base_class,
                features_train=support_x,
                labels_train=one_hot_labels)
            label_known = query_y[:args.meta_test_way * args.meta_test_query]
            pred_score, pred = torch.max(cls_scores.data, 1)
            pred_known = pred[:args.meta_test_way * args.meta_test_query]
            total_correct += pred_known.eq(label_known).sum().item()
            total_num += args.meta_test_way * args.meta_test_query
            acc.append(pred_known.eq(label_known).sum().item() / (args.meta_test_way * args.meta_test_query))
            score_known = pred_score[:args.meta_test_way * args.meta_test_query]
            score_unknown = pred_score[args.meta_test_way * args.meta_test_query:]
            correct_idx = torch.where(pred_known == label_known)
            score_correct = score_known[correct_idx]
            l = np.concatenate((np.array([1] * args.meta_test_way * args.meta_test_query),
                                np.array([0] * args.test_oway * args.test_oquery)))
            all_label.append(l)
            all_dist.append(np.concatenate((score_known.cpu().numpy(), score_unknown.cpu().numpy())))
            unknown_dist.append(score_unknown.cpu().numpy())
            correct_dist.append(score_correct.cpu().numpy())

        print("########################auroc###########################")
        auroc = []
        for i in warp_tqdm(range(len(all_dist))):
            fpr, tpr, thresholds = metrics.roc_curve(all_label[i], all_dist[i], pos_label=1)
            auroc.append(metrics.auc(fpr, tpr))


        auroc_m, auroc_pm = compute_confidence_interval(auroc)
        acc_m, acc_pm = compute_confidence_interval(acc)
    return acc_m, acc_pm, auroc_m, auroc_pm


def save_stage1(model):
    '''
    get the saved feature
    input:
        model:             the pretrained model
    output:
        train_mean:        mean of all features in the training set  
        train_weight:      mean of each class in the training set
        train_sample:      features of each sample in the training set
        val_sample:        features of each sample in the val set
        test_sample:       features of each sample in the test set
    '''
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    val_loader = get_dataloader('val', aug=False, shuffle=False, out_name=False)
    test_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)

    train_mean, train_weight, train_sample, val_sample, test_sample = extract_feature(train_loader, val_loader,
                                                                                      test_loader, model)
    return train_mean, train_weight, train_sample, val_sample, test_sample


def extract_feature(train_loader, val_loader, test_loader, model):
    '''
    forward the image to model to get feature and save it
    input:
        train_loader:
        val_loader:
        test_loader: 
        model:
    output:
        train_mean:        mean of all features in the training set  
        train_weight:      mean of each class in the training set
        train_sample:      features of each sample in the training set
        val_sample:        features of each sample in the val set
        test_sample:       features of each sample in the test set
    '''
    save_dir = args.save_path
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # get training mean
        train_mean = []
        train_sample = collections.defaultdict(list)
        train_weight = np.zeros((args.num_classes, 512))
        for i, (inputs, labels) in enumerate(warp_tqdm(train_loader)):
            outputs = model(inputs.cuda(args.gpu))
            outputs = outputs.cpu().data.numpy()
            train_mean.append(outputs)
            for out, label in zip(outputs, labels):
                train_sample[label.item()].append(out)
        for i in range(args.num_classes):
            train_weight[i] = np.array(train_sample[i]).mean(0)
        train_mean = np.concatenate(train_mean, axis=0).mean(0)

        val_sample = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
            # compute output
            outputs = model(inputs.cuda(args.gpu))
            outputs = outputs.cpu().data.numpy()
            for out, label in zip(outputs, labels):
                val_sample[label.item()].append(out)

        test_sample = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(test_loader)):
            # compute output
            outputs = model(inputs.cuda(args.gpu))
            outputs = outputs.cpu().data.numpy()
            for out, label in zip(outputs, labels):
                test_sample[label.item()].append(out)

        all_info = [train_mean, train_weight, train_sample, val_sample, test_sample]
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info


def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None):
    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(args.image_size, disable_random_resize=args.disable_random_resize)
    else:
        transform = datasets.without_augment(args.image_size, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader


def get_optimizer(module):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]


def get_scheduler(batches, optimiter):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, args.lr_stepsize, args.lr_gamma),
                 'multi_step': MultiStepLR(optimiter, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)],
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.epochs, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/{}_'.format(epoch) + filename)
    if is_best:
        shutil.copyfile(folder + '/{}_'.format(epoch) + filename, folder + '/model_best.pth.tar')


def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger

def load_checkpoint(model, path):
    weights = torch.load(path)
    model_weights = weights['params']
    new_weights = OrderedDict()
    for key in model_weights.keys():  # module.
        if key[:7] == 'encoder':
            name = key[8:]
            new_weights[name] = model_weights[key]
        else:
            name = key
            new_weights[name] = model_weights[key]
    missing_keys, unexpected_keys = model.load_state_dict(new_weights, strict=False)
    print('********************missing_keys*************************')
    print(missing_keys)
    print('********************unexpected_keys*************************')
    print(unexpected_keys)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


if __name__ == '__main__':
    main()