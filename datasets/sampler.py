import numpy as np
import torch
from torch.utils.data import Sampler
import torch.nn.functional as F

__all__ = ['CategoriesSampler']

'''class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query, selective_bag=None):
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.selective_bag = selective_bag
        self.chosen_class = np.zeros(64)
        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            if self.selective_bag is None:
                classes = torch.randperm(len(self.m_ind))[:self.n_way]
                ukclass = classes[5:]
                for uc in ukclass:
                    self.chosen_class[uc] += 1
            else:
                classes = torch.randperm(len(self.m_ind))[:self.n_way//2]
                unknown_map, chosen_class = self.selective_bag
                for c in classes:
                    opensample = unknown_map[c.item()]
                    for o in opensample:
                        o = o.item()
                        o = torch.IntTensor(1).fill_(o)
                        if o not in classes :#and self.chosen_class[o] <= 3000
                            o = o.view(1)
                            classes = torch.cat((classes, o), 0)
                            self.chosen_class[o] += 1
                            break
                    else:
                        for c in chosen_class:
                            c = torch.IntTensor(1).fill_(c)
                            if c not in classes and self.chosen_class[c] <= 3000:
                                c = c.view(1)
                                classes = torch.cat((classes, c), 0)
                                self.chosen_class[c] += 1
                                break
            #print(classes, self.chosen_class)
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])

            batch = torch.cat(batch_gallery + batch_query)
            yield batch'''


class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_gallery + batch_query)
            yield batch
