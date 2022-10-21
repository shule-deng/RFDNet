from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def stage2_episode(sample, way, shot, query, do_open=False, oway=None, oshot=None, oquery=None):
    '''
    creat a task with both closed episode and open episode
    input:
        sample:         feature of the dataset used for the creat a task
        way:            way
        shot:           shot
        query:          query
        do_open:        Whether to sample open episode
        oway:           way for open episode 
        oshot:          shot for open episode
        oquery:         query for open episode
    output:
        total_x:        samples of current task
        total_y:        label of current task 
        rest_class:     base class of current task 
    '''
    total_class = np.sort(np.array(list(sample.keys())))
    known_class = np.random.choice(total_class, way, False)
    rest_class = np.setdiff1d(total_class, known_class)

    support_known_x = []
    support_konwn_y = []
    query_known_x = []
    query_konwn_y = []

    for class_id in known_class:
        sample_num = len(sample[class_id])
        chosen_id = np.random.choice(sample_num, shot + query, False)
        support_id = np.array(chosen_id[:shot])
        query_id = np.array(chosen_id[shot:])

        support_known_x.append(np.array(sample[class_id])[support_id])
        support_konwn_y.extend([class_id] * shot)
        query_known_x.append(np.array(sample[class_id])[query_id])
        query_konwn_y.extend([class_id] * query)
    support_known_x = np.array(support_known_x).reshape(-1, 512)
    query_known_x = np.array(query_known_x).reshape(-1, 512)

    total_x = np.concatenate((support_known_x, query_known_x), axis=0)
    total_y = support_konwn_y + query_konwn_y

    if do_open:
        query_unknown_x = []
        query_unkonwn_y = []
        unknown_class = np.random.choice(rest_class, oway, False)
        rest_class = np.setdiff1d(rest_class, unknown_class)
        for unclass_id in unknown_class:
            sample_num = len(sample[unclass_id])
            chosen_id = np.random.choice(sample_num, oquery, False)
            query_unknown_x.append(np.array(sample[unclass_id])[chosen_id])
            query_unkonwn_y.extend([unclass_id] * oquery)
        query_unknown_x = np.array(query_unknown_x).reshape(-1, 512)
        total_x = np.concatenate((total_x, query_unknown_x), axis=0)
        total_y = total_y + query_unkonwn_y

    assert total_x.shape[0] == len(total_y), "episode error"
    return total_x, total_y, rest_class


class FeatExemplarAvgBlock(nn.Module):
    '''
    get prototypes for each category
    '''
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        '''
        input:
            features_train:     feature of support
            labels_train:       label of support
        output:
            weight_novel:       prototype
        '''
        labels_train_transposed = labels_train.transpose(1, 2)
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel


class AttentionBasedBlock(nn.Module):
    '''
    obtain original displacement by weighted summation of the base class centers
    '''
    def __init__(self, nFeat, nK, weight_base, scale_att=10.0):
        '''
        input:
            nFeat:             feature dimension 
            nK:                the number of classes of the base class
            weight_base:       base class centers
            scale_att:         learnable scaling factor of cosine similarity
        '''
        super(AttentionBasedBlock, self).__init__()
        self.nFeat = nFeat
        self.scale_att = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_att), requires_grad=True)
        wkeys = torch.FloatTensor(nK, nFeat).normal_(0.0, np.sqrt(2.0 / nFeat))
        self.wkeys = nn.Parameter(wkeys, requires_grad=True)


    def forward(self, features_train, weight_base, Kbase):
        '''
        input:
            features_train:    input feature
            weight_base:       base class centers
            Kbase:             base class idx
        output:
            weight_novel:      original displacement
        '''
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1)

        features_train = features_train.view(batch_size * num_train_examples, num_features)
        Qe = features_train
        Qe = Qe.view(batch_size, num_train_examples, self.nFeat)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim() - 1, eps=1e-12)

        wkeys = self.wkeys
        wkeys = wkeys[Kbase.view(-1)]  # the keys of the base categoreis
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim() - 1, eps=1e-12)
        wkeys = wkeys.view(batch_size, nKbase, self.nFeat).transpose(1, 2)

        AttentionCoeficients = self.scale_att *torch.bmm(Qe, wkeys)  #
        AttentionCoeficients = F.softmax(AttentionCoeficients.view(batch_size * num_train_examples, nKbase), dim=1)
        AttentionCoeficients = AttentionCoeficients.view(batch_size, num_train_examples, nKbase)

        weight_novel = torch.bmm(AttentionCoeficients, weight_base)

        return weight_novel


class wgenerator(nn.Module):
    '''
    generate task-adaptive weights for displacement
    '''
    def __init__(self, nFeat):
        super(wgenerator, self).__init__()
        self.fc1 = nn.Linear(nFeat, nFeat)
        self.fc1.weight.data.copy_(torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat)*0.001)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(nFeat, nFeat)
        self.fc2.weight.data.copy_(torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat)*0.001)
        self.fc2.bias.data.zero_()


    def forward(self, support):
        '''
        input:
            support:  support set
        output:
            weight:   task-adaptive weights for displacement
        '''
        way, dim = support.size(0), support.size(1)
        weight= []
        all_id = np.array(list(range(way)))
        for c in range(way):
            rest_id = np.setdiff1d(all_id, c)
            p = support[c]
            restp = support[rest_id]
            difference = torch.abs(p - restp)
            difference = difference.mean(0)
            weighti = self.fc1(difference)
            weighti = self.fc2(weighti)
            weighti = weighti.view(1, -1)
            weight.append(weighti)
        weight = torch.cat(weight, 0)
        return weight


class Classifier(nn.Module):
    '''
    the TRFD module
    get_classification_weights:    
        1. generate prototype for novel class
        2. generate task-adaptive weights for displacement
        3. obtain original displacement for support
        4. apply the task-adaptive weights to the displacement of support
        5. add the displacement to the original feature
    apply_classification_weights:
        1. obtain original displacement for query
        2. apply the task-adaptive weights to the displacement of query
        3. add the displacement to the original feature
        4. calculate the similarity
    '''
    def __init__(self, args, weight_base):
        super(Classifier, self).__init__()
        self.args = args
        self.weight_generator_type = args.weight_generator_type
        self.classifier_type = args.classifier_type
        assert (self.classifier_type == 'cosine' or
                self.classifier_type == 'dotproduct')
        nKall = args.num_classes
        nFeat = args.out_dim
        self.nFeat = nFeat
        self.nKall = nKall

        weight_base = torch.FloatTensor(weight_base).cuda(args.gpu)
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)

        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        scale_cls = args.scale_cls
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)

        if self.weight_generator_type == 'attention_based':
            scale_att = args.scale_cls
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.attblock = AttentionBasedBlock(
                nFeat, nKall, self.weight_base, scale_att=scale_att)

        else:
            raise ValueError('Not supported/recognized type {0}'.format(
                self.weight_generator_type))

        self.weight_generator = wgenerator(self.nFeat)

    def get_classification_weights(
            self, Kbase_ids, features_train=None, labels_train=None):
        '''
        input:
            Kbase_ids:          base class ids  
            features_train:     support feature
            labels_train:       support label
        output:
            weight_both:        the final prototype 
        '''
        # ***********************************************************************
        # ******** Get the classification weights for the base categories *******
        batch_size, nKbase = Kbase_ids.size()
        weight_base = self.weight_base[Kbase_ids.view(-1)]
        weight_base = weight_base.view(batch_size, nKbase, -1)
        # ***********************************************************************

        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        # ***********************************************************************
        # ******* Generate classification weights for the novel categories ******
        _, num_train_examples, num_channels = features_train.size()
        nKnovel = labels_train.size(2)
        if self.classifier_type == 'cosine':
            features_train = F.normalize(
                features_train, p=2, dim=features_train.dim() - 1, eps=1e-12)
        if self.weight_generator_type == 'attention_based':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel_avg = weight_novel_avg.view(batch_size * nKnovel, num_channels)
            self.weight = self.weight_generator(weight_novel_avg)
            self.weight = self.weight.cuda(self.args.gpu)

            if self.classifier_type == 'cosine':
                weight_base_tmp = F.normalize(
                    weight_base, p=2, dim=weight_base.dim() - 1, eps=1e-12)
            else:
                weight_base_tmp = weight_base

            weight_novel_att = self.attblock(weight_novel_avg.view(1, nKnovel, num_channels), weight_base_tmp,
                                             Kbase_ids)
            weight_novel_att = weight_novel_att * self.weight
            weight_novel_att = weight_novel_att.view(weight_novel_att.size(1), weight_novel_att.size(2))
            weight_novel = weight_novel_avg + weight_novel_att
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)

        else:
            raise ValueError('Not supported / recognized type {0}'.format(
                self.weight_generator_type))

        weight_both = torch.cat([weight_base, weight_novel], dim=1)

        return weight_both

    def apply_classification_weights(self, features, cls_weights, Kbase_ids):
        '''
        input:
            features:       feature of query
            cls_weights:    prototype
            Kbase_ids:      base class ids
        output:
            score:          the similarity score between query and each prototype
        '''
        features = F.normalize(features, p=2, dim=features.dim() - 1, eps=1e-12)
        if self.weight_generator_type == 'attention_based':
            batch_size, nKbase = Kbase_ids.size()

            weight_base = self.weight_base[Kbase_ids.view(-1)]
            weight_base = weight_base.view(batch_size, nKbase, -1)
            weight_novel_avg = features.view(features.size(1), features.size(2))

            if self.classifier_type == 'cosine':
                weight_base_tmp = F.normalize(
                    weight_base, p=2, dim=weight_base.dim() - 1, eps=1e-12)
            else:
                weight_base_tmp = weight_base

            weight_novel_att = self.attblock(features, weight_base_tmp, Kbase_ids)
            weight_novel_att = weight_novel_att.view(weight_novel_att.size(1), weight_novel_att.size(2))
            cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12)
            way = self.weight.size(0)
            score = torch.zeros(features.size(1), way).cuda(self.args.gpu)
            for i in range(way):
                weighti = self.weight[i]
                weight_novel_atti = weight_novel_att * weighti
                features = weight_novel_avg + weight_novel_atti
                features = F.normalize(features, p=2, dim=features.dim() - 1, eps=1e-12)
                features = features.view(1, features.size(0), features.size(1))
                cls_scores = self.scale_cls * torch.baddbmm(self.bias.view(1, 1, 1), features, cls_weights.transpose(1, 2))
                score[:, i] = cls_scores[0, :, -way + i]
            return score

    def forward(self, features_test, Kbase_ids, features_train=None, labels_train=None):
        '''
        input:
            features_test:      query feature
            Kbase_ids:          base class ids
            features_train:     support feature
            labels_train:       supporet label
        output:
            cls_scores:         the similarity score between query and each prototype
        '''
        cls_weights = self.get_classification_weights(
            Kbase_ids, features_train, labels_train)
        cls_scores = self.apply_classification_weights(
            features_test, cls_weights, Kbase_ids)
        return cls_scores



class open_loss(nn.Module):
    '''
    MOS loss
    '''
    def __init__(self, lamda, way, query, oway, oquery, gpu):
        super(open_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lamda = lamda
        self.way = way
        self.query = query
        self.oway = oway
        self.oquery = oquery
        self.gpu = gpu

    def forward(self, x, y):
        '''
        input:
            x:          predicted results
            y:          label
        output:
            total_loss: the MOS loss value    
        '''
        known_x = x[:self.way * self.query, :]
        known_y = y[:self.way * self.query]
        unknown_x = x[self.way * self.query:, :]
        unknown_y = y[self.way * self.query:]

        loss_known1 = self.ce(known_x, known_y)

        one_hot_labels = torch.zeros(self.way * self.query, self.way).cuda(device=self.gpu).scatter_(1, known_y.view(-1, 1),1)
        dim1 = torch.nonzero(one_hot_labels)
        known_x_gt = known_x[dim1[:, 0], dim1[:, 1]]
        loss_known2 = 2 - known_x_gt
        loss_known2 = torch.clamp(loss_known2, 0)
        loss_known2 = loss_known2.mean()

        loss_unknown2 = unknown_x + 2
        loss_unknown2 = torch.clamp(loss_unknown2, 0)
        loss_unknown2 = loss_unknown2.mean()

        total_loss = self.lamda * (loss_known2 + loss_unknown2) + loss_known1
        return total_loss










