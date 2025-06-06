import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pdb
import csv

def load_labels(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=mean, std=std)

class PreTiered(Dataset):
    def __init__(self, args, partition='train', is_training=True, is_contrast=False):
        super(PreTiered, self).__init__()
        self.is_contrast = is_training and is_contrast

        if is_training:
            if is_contrast:
                self.transform_left = transforms.Compose([
                    transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize
                ])
                self.transform_right = transforms.Compose([
                    transforms.RandomRotation(10),
                    transforms.RandomCrop(84, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),normalize])
        
        #image_file = '{}_images.npz'.format(partition)
        #label_file = '{}_labels.pkl'.format(partition)
        image_file = 'few-shot-{}.npz'.format(partition)

        # modified code to load tieredImageNet
        image_file = os.path.join(args.data_root, image_file)
        self.imgs = np.load(image_file)['features']
        labels = np.load(image_file)['targets'].astype(np.int64)
        idx_to_class = {}
        with open('/root/wjg/jbw/SEMOP/SemFew/find_labels.csv', mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                idx = int(row[0])
                class_name = row[1]
                idx_to_class[idx] = class_name
        self.idx_to_class = idx_to_class
        # label_file = os.path.join(args.data_root, label_file)
        # labels = load_labels(label_file)['labels']

        self.imgs = [Image.fromarray(x) for x in self.imgs]
        min_label = min(labels)
        self.min_label = min_label
        if partition=='val':
            self.min_label = 351
        if partition=='test':
            self.min_label = 448
        self.labels = [x - min_label for x in labels]
        print('Load {} Data of {} for tieredImageNet in Pretraining Stage'.format(len(self.imgs), partition))

    def __getitem__(self, item):
        idx_to_class=self.idx_to_class
        min_label = self.min_label
        if self.is_contrast:
            left,right = self.transform_left(self.imgs[item]),self.transform_right(self.imgs[item])
            target = self.labels[item]
            return left, right, target, item,idx_to_class,min_label
        else:
            img = self.transform(self.imgs[item])
            target = self.labels[item]
            return img, target, item,idx_to_class,min_label
        
    def __len__(self):
        return len(self.labels)

class MetaTiered(Dataset):
    def __init__(self, args, n_shots, partition='test', is_training=False, fix_seed=True):
        super(MetaTiered, self).__init__()
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = n_shots
        self.n_queries = args.n_queries
        self.n_episodes = args.n_episodes
        self.n_aug_support_samples = args.n_aug_support_samples

        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        if is_training:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),normalize])
        # data_sum = 'few-shot-{}.npz'.format(partition)
        # data_sum = os.path.join(args.data_root, data_sum)
        # self.imgs = np.load(data_sum)
        # test = self.imgs['features']

        image_file = 'few-shot-{}.npz'.format(partition)
        #label_file = '{}_labels.pkl'.format(partition)

        # modified code to load tieredImageNet
        image_file = os.path.join(args.data_root, image_file)
        self.imgs = np.load(image_file)['features']
        labels = np.load(image_file)['targets'].astype(np.int64)
        idx_to_class = {}
        with open('/root/wjg/jbw/SEMOP/SemFew/find_labels.csv', mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                idx = int(row[0])
                class_name = row[1]
                idx_to_class[idx] = class_name
        self.idx_to_class = idx_to_class
        # label_file = os.path.join(args.data_root, image_file)
        # labels = load_labels(label_file)['targets']

        self.imgs = [Image.fromarray(x) for x in self.imgs]
        min_label = min(labels)
        self.labels = [x - min_label for x in labels]
        self.min_label = min_label
        if partition=='val':
            self.min_label = 351
        if partition=='test':
            self.min_label = 448
        print('Load {} Data of {} for tieredImageNet in Meta-Learning Stage'.format(len(self.imgs), partition))
        
        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())        
    
    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * query_xs_ids.shape[0])

        if self.n_aug_support_samples > 1:
            support_xs = support_xs * self.n_aug_support_samples 
            support_ys = support_ys * self.n_aug_support_samples 
        
        support_xs = torch.stack(list(map(lambda x: self.train_transform(x), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x), query_xs)))
        support_ys,query_ys = np.array(support_ys),np.array(query_ys)
        idx_to_class = self.idx_to_class
        min_label =self.min_label
        cls_sampled= np.array(cls_sampled)
        return support_xs, support_ys, query_xs, query_ys,cls_sampled,idx_to_class,min_label
    
    def __len__(self):
        return self.n_episodes