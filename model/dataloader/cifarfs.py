import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
#IMAGE_PATH1 = osp.join(ROOT_PATH, 'data/miniimagenet/images')
#SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
#CACHE_PATH = osp.join(ROOT_PATH, 'cache/')

def identity(x):
    return x
    

class CIFARFS(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        
        
        self.IMAGE_PATH1 = osp.join(args.data_path, 'data')
        #self.SPLIT_PATH = osp.join(args.split_path, 'split')
        self.SPLIT_PATH = args.split_path
        
        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        
        self.data, self.label = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))
        
        
        image_size = 32
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(40),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])                    
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
    
            
    def parse_csv(self, csv_path, setname):
        with open(csv_path, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']
        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        return data, mapped_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        
        image = Image.open(self.IMAGE_PATH1 + '/' + self.data[i]).convert('RGB')
        image = self.transform(image)
        label = int(label)
        
        return image, label

