import os
import pickle
from PIL import Image
import numpy as np
import torch
import pdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json

class OpenMini(Dataset):
    def __init__(self, args, partition='test', mode='episode', is_training=False, fix_seed=True):
        super(OpenMini, self).__init__()
        self.mode = mode
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_open_ways = args.n_open_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_episodes = args.n_test_runs if partition == 'test' else args.n_train_runs
        self.n_aug_support_samples = 1 if partition == 'train' else args.n_aug_support_samples#这里在test时增强样本
        self.partition = partition
        
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

        with open(os.path.join(args.data_root,'miniImageNet_category_vector.pickle'), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        vector_array = []
        for i in range(100):
            vector_array.append(pack[i][1])
        vector_array = np.array(vector_array) # Train 0~63, Val 64~79, Test 80~99
        self.vector_array = {'base':vector_array[:64],'nove_val':vector_array[64:80],'novel_test':vector_array[80:]}
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),normalize])
        self.init_episode(args.data_root,partition)
        # self.get_episode(10)        
    
    def __getitem__(self, item):
        return self.get_episode(item)

    def init_episode(self, data_root, partition):
        suffix = partition if partition in ['val','test'] else 'train_phase_train'
        filename = 'miniImageNet_category_split_{}.pickle'.format(suffix)         
        self.data = {}
        
        with open(os.path.join(data_root, filename), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        self.idx_to_class = {k: v for v, k in pack['catname2label'].items()}
        #print(idx_to_class)
        '''
        {52: 'n04509417', 59: 'n07697537', 41: 'n03924679', 16: 'n02113712', 8: 'n02089867', 13: 'n02108551', 39: 'n03888605', 50: 'n04435653', 7: 'n02074367', 26: 'n03017168', 24: 'n02823428', 31: 'n03337140', 21: 'n02687172', 12: 'n02108089', 63: 'n13133613', 2: 'n01704323', 18: 'n02165456', 0: 'n01532829', 14: 'n02108915', 6: 'n01910747', 30: 'n03220513', 27: 'n03047690', 19: 'n02457408', 5: 'n01843383', 22: 'n02747177', 3: 'n01749939', 62: 'n13054560', 29: 'n03207743', 11: 'n02105505', 10: 'n02101006', 61: 'n09246464', 45: 'n04251144', 49: 'n04389033', 32: 'n03347037', 43: 'n04067472', 1: 'n01558993', 58: 'n07584110', 28: 'n03062245', 53: 'n04515003', 42: 'n03998194', 46: 'n04258138', 57: 'n06794110', 25: 'n02966193', 23: 'n02795169', 34: 'n03476684', 33: 'n03400231', 4: 'n01770081', 17: 'n02120079', 35: 'n03527444', 60: 'n07747607', 51: 'n04443257', 47: 'n04275548', 54: 'n04596742', 20: 'n02606052', 56: 'n04612504', 55: 'n04604644', 44: 'n04243546', 36: 'n03676483', 38: 'n03854065', 15: 'n02111277', 48: 'n04296562', 40: 'n03908618', 9: 'n02091831', 37: 'n03838899'}
        '''
        imgs = pack['data'].astype('uint8')
        labels = pack['labels']
        self.imgs = [Image.fromarray(x) for x in imgs]
        self.min_label = min(labels)
        #print('{}:'.format(partition),min_label)#train: 0 #test: 80
        self.labels = [x - self.min_label for x in labels]
        print('Load {} Data of {} for miniImagenet in Meta-Learning Stage'.format(len(self.imgs), partition))
        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    
    def get_episode(self, item):
        
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        suppopen_xs = []
        suppopen_ys = []
        query_xs = []
        query_ys = []
        openset_xs = []
        openset_ys = []
        manyshot_xs = []
        manyshot_ys = []

        # Close set preparation
        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * self.n_queries)
        
        # Open set preparation
        cls_open_ids = np.setxor1d(np.arange(len(self.classes)), cls_sampled)
        cls_open_ids = np.random.choice(cls_open_ids, self.n_open_ways, False)
        for idx, the_cls in enumerate(cls_open_ids):
            imgs = self.data[the_cls]
            suppopen_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            suppopen_xs.extend([imgs[the_id] for the_id in suppopen_xs_ids_sampled])
            suppopen_ys.extend([idx] * self.n_shots)
            openset_xs_ids = np.setxor1d(np.arange(len(imgs)), suppopen_xs_ids_sampled)
            openset_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_queries, False)
            openset_xs.extend([imgs[the_id] for the_id in openset_xs_ids_sampled])
            openset_ys.extend([the_cls] * self.n_queries)

        if self.partition == 'train':
            base_ids = np.setxor1d(np.arange(len(self.classes)), np.concatenate([cls_sampled,cls_open_ids]))
            assert len(set(base_ids).union(set(cls_open_ids)).union(set(cls_sampled))) == 64
            base_ids = np.array(sorted(base_ids))


        if self.n_aug_support_samples > 1:
            support_xs_aug = [support_xs[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_xs),self.n_shots)]
            support_ys_aug = [support_ys[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_ys),self.n_shots)]
            support_xs,support_ys = support_xs_aug[0],support_ys_aug[0]
            for next_xs,next_ys in zip(support_xs_aug[1:],support_ys_aug[1:]):
                support_xs.extend(next_xs)
                support_ys.extend(next_ys)

            suppopen_xs_aug = [suppopen_xs[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_xs),self.n_shots)]
            suppopen_ys_aug = [suppopen_ys[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_ys),self.n_shots)]
            suppopen_xs,suppopen_ys = suppopen_xs_aug[0],suppopen_ys_aug[0]
            for next_xs,next_ys in zip(suppopen_xs_aug[1:],suppopen_ys_aug[1:]):
                suppopen_xs.extend(next_xs)
                suppopen_ys.extend(next_ys)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x), support_xs)))
        suppopen_xs =  torch.stack(list(map(lambda x: self.train_transform(x), suppopen_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x), query_xs)))
        openset_xs = torch.stack(list(map(lambda x: self.test_transform(x), openset_xs)))
        support_ys,query_ys,openset_ys = np.array(support_ys),np.array(query_ys),np.array(openset_ys)
        suppopen_ys = np.array(suppopen_ys)
        cls_sampled, cls_open_ids = np.array(cls_sampled), np.array(cls_open_ids)
        idx_to_class = self.idx_to_class
        min_label=self.min_label
        #print("idx")
        #print(idx_to_class)
        #if self.partition=='test':
        #   print("test-support_xs",support_xs.shape)#torch.Size([125, 3, 84, 84]) 
        '''
        此处包含了两组,为batchsize设置
        {52: 'n04509417', 59: 'n07697537', 41: 'n03924679', 16: 'n02113712', 8: 'n02089867', 13: 'n02108551', 39: 'n03888605', 50: 'n04435653', 7: 'n02074367', 26: 'n03017168', 24: 'n02823428', 31: 'n03337140', 21: 'n02687172', 12: 'n02108089', 63: 'n13133613', 2: 'n01704323', 18: 'n02165456', 0: 'n01532829', 14: 'n02108915', 6: 'n01910747', 30: 'n03220513', 27: 'n03047690', 19: 'n02457408', 5: 'n01843383', 22: 'n02747177', 3: 'n01749939', 62: 'n13054560', 29: 'n03207743', 11: 'n02105505', 10: 'n02101006', 61: 'n09246464', 45: 'n04251144', 49: 'n04389033', 32: 'n03347037', 43: 'n04067472', 1: 'n01558993', 58: 'n07584110', 28: 'n03062245', 53: 'n04515003', 42: 'n03998194', 46: 'n04258138', 57: 'n06794110', 25: 'n02966193', 23: 'n02795169', 34: 'n03476684', 33: 'n03400231', 4: 'n01770081', 17: 'n02120079', 35: 'n03527444', 60: 'n07747607', 51: 'n04443257', 47: 'n04275548', 54: 'n04596742', 20: 'n02606052', 56: 'n04612504', 55: 'n04604644', 44: 'n04243546', 36: 'n03676483', 38: 'n03854065', 15: 'n02111277', 48: 'n04296562', 40: 'n03908618', 9: 'n02091831', 37: 'n03838899'}
        {52: 'n04509417', 59: 'n07697537', 41: 'n03924679', 16: 'n02113712', 8: 'n02089867', 13: 'n02108551', 39: 'n03888605', 50: 'n04435653', 7: 'n02074367', 26: 'n03017168', 24: 'n02823428', 31: 'n03337140', 21: 'n02687172', 12: 'n02108089', 63: 'n13133613', 2: 'n01704323', 18: 'n02165456', 0: 'n01532829', 14: 'n02108915', 6: 'n01910747', 30: 'n03220513', 27: 'n03047690', 19: 'n02457408', 5: 'n01843383', 22: 'n02747177', 3: 'n01749939', 62: 'n13054560', 29: 'n03207743', 11: 'n02105505', 10: 'n02101006', 61: 'n09246464', 45: 'n04251144', 49: 'n04389033', 32: 'n03347037', 43: 'n04067472', 1: 'n01558993', 58: 'n07584110', 28: 'n03062245', 53: 'n04515003', 42: 'n03998194', 46: 'n04258138', 57: 'n06794110', 25: 'n02966193', 23: 'n02795169', 34: 'n03476684', 33: 'n03400231', 4: 'n01770081', 17: 'n02120079', 35: 'n03527444', 60: 'n07747607', 51: 'n04443257', 47: 'n04275548', 54: 'n04596742', 20: 'n02606052', 56: 'n04612504', 55: 'n04604644', 44: 'n04243546', 36: 'n03676483', 38: 'n03854065', 15: 'n02111277', 48: 'n04296562', 40: 'n03908618', 9: 'n02091831', 37: 'n03838899'}'''
        if self.partition == 'train':
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, cls_sampled, cls_open_ids, base_ids, idx_to_class,min_label
        else:
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, cls_sampled, cls_open_ids,idx_to_class,min_label

        
    def __len__(self):
        return self.n_episodes

  





class GenMini(OpenMini):
    def __init__(self, args, partition='test', mode='episode', is_training=False, fix_seed=True):
        super(GenMini, self).__init__(args, partition, mode, is_training, fix_seed)

    def __getitem__(self, item):
        return self.get_episode(item)
        
    def init_episode(self, data_root, partition):

        if partition == 'train':

            filename = 'miniImageNet_category_split_train_phase_train.pickle'
            with open(os.path.join(data_root, filename), 'rb') as f:
                pack = pickle.load(f, encoding='latin1')
            self.base_imgs = pack['data'].astype('uint8')
            labels = pack['labels']
            self.base_imgs = [Image.fromarray(x) for x in self.base_imgs]
            min_label = min(labels)
            self.base_labels = [x - min_label for x in labels]
            self.base_data = {}
            for idx in range(len(self.base_imgs)):
                if self.base_labels[idx] not in self.base_data:
                    self.base_data[self.base_labels[idx]] = []
                self.base_data[self.base_labels[idx]].append(self.base_imgs[idx])
            self.base_classes = list(self.base_data.keys())

            self.novel_imgs = self.base_imgs
            self.novel_labels = self.base_labels
            self.novel_data = self.base_data
            self.novel_classes = self.base_classes
        
        elif partition == 'test':

            filename = 'miniImageNet_category_split_train_phase_test.pickle'
            with open(os.path.join(data_root, filename), 'rb') as f:
                pack = pickle.load(f, encoding='latin1')
            self.base_imgs = pack['data'].astype('uint8')
            labels = pack['labels']
            self.base_imgs = [Image.fromarray(x) for x in self.base_imgs]
            min_label = min(labels)
            self.base_labels = [x - min_label for x in labels]
            self.base_data = {}
            for idx in range(len(self.base_imgs)):
                if self.base_labels[idx] not in self.base_data:
                    self.base_data[self.base_labels[idx]] = []
                self.base_data[self.base_labels[idx]].append(self.base_imgs[idx])
            self.base_classes = list(self.base_data.keys())

            filename = 'miniImageNet_category_split_test.pickle'
            with open(os.path.join(data_root, filename), 'rb') as f:
                pack = pickle.load(f, encoding='latin1')
            self.novel_imgs = pack['data'].astype('uint8')
            labels = pack['labels']
            self.novel_imgs = [Image.fromarray(x) for x in self.novel_imgs]
            min_label = min(labels)
            self.novel_labels = [x - min_label + len(self.base_classes) for x in labels]
            self.novel_data = {}
            for idx in range(len(self.novel_imgs)):
                if self.novel_labels[idx] not in self.novel_data:
                    self.novel_data[self.novel_labels[idx]] = []
                self.novel_data[self.novel_labels[idx]].append(self.novel_imgs[idx])
            self.novel_classes = list(self.novel_data.keys())

        print('Load {} Data of {} for miniImagenet in Meta-Learning Stage'.format(len(self.base_imgs), partition))
        print('Load {} Data of {} for miniImagenet in Meta-Learning Stage'.format(len(self.novel_imgs), partition))
    
    def get_episode(self, item):
        
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.novel_classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        suppopen_xs = []
        suppopen_ys  = []
        openset_xs = []
        openset_ys = []
        manyshot_xs = []
        manyshot_ys = []

        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.novel_data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * self.n_queries)
        
        cls_open_ids = np.setxor1d(self.novel_classes, cls_sampled)
        cls_open_ids = np.random.choice(cls_open_ids, self.n_ways, False)
        for idx, the_cls in enumerate(cls_open_ids):
            imgs = self.novel_data[the_cls]
            suppopen_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            suppopen_xs.extend([imgs[the_id] for the_id in suppopen_xs_ids_sampled])
            suppopen_ys.extend([idx] * self.n_shots)
            openset_xs_ids = np.setxor1d(np.arange(len(imgs)), suppopen_xs_ids_sampled)
            openset_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_queries, False)
            openset_xs.extend([imgs[the_id] for the_id in openset_xs_ids_sampled])
            openset_ys.extend([the_cls] * self.n_queries)
        
        if self.partition == 'train':
            base_ids = np.setxor1d(self.base_classes, np.concatenate([cls_sampled,cls_open_ids]))
            assert len(set(base_ids).union(set(cls_open_ids)).union(set(cls_sampled))) == 64
            base_ids = sorted(base_ids)
        else:
            base_ids = sorted(self.base_classes)
        
        num_query = self.n_ways * self.n_queries
        assert num_query > len(base_ids)
        num_atleast = num_query//len(base_ids)
        num_extra = list(np.random.choice(base_ids, num_query-len(base_ids)*num_atleast, False))
        num_extra.sort()
        num_samples = {}
        for the_cls in base_ids:
            num_samples[the_cls] = num_atleast + 1 if the_cls in num_extra else num_atleast

        for idx, the_cls in enumerate(base_ids):
            imgs = self.base_data[the_cls]
            manyshot_xs_ids_sampled = np.random.choice(range(len(imgs)), num_samples[the_cls], False)
            manyshot_xs.extend([imgs[the_id] for the_id in manyshot_xs_ids_sampled])
            manyshot_ys.extend([idx] * num_samples[the_cls])
            
        if self.n_aug_support_samples > 1:
            support_xs_aug = [support_xs[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_xs),self.n_shots)]
            support_ys_aug = [support_ys[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_ys),self.n_shots)]
            support_xs,support_ys = support_xs_aug[0],support_ys_aug[0]
            for next_xs,next_ys in zip(support_xs_aug[1:],support_ys_aug[1:]):
                support_xs.extend(next_xs)
                support_ys.extend(next_ys)
            
            suppopen_xs_aug = [suppopen_xs[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_xs),self.n_shots)]
            suppopen_ys_aug = [suppopen_ys[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_ys),self.n_shots)]
            suppopen_xs,suppopen_ys = suppopen_xs_aug[0],suppopen_ys_aug[0]
            for next_xs,next_ys in zip(suppopen_xs_aug[1:],suppopen_ys_aug[1:]):
                suppopen_xs.extend(next_xs)
                suppopen_ys.extend(next_ys)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x), support_xs)))
        suppopen_xs =  torch.stack(list(map(lambda x: self.train_transform(x), suppopen_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x), query_xs)))
        openset_xs = torch.stack(list(map(lambda x: self.test_transform(x), openset_xs)))
        manyshot_xs = torch.stack(list(map(lambda x: self.test_transform(x), manyshot_xs)))
        support_ys,query_ys,openset_ys = np.array(support_ys),np.array(query_ys),np.array(openset_ys)
        suppopen_ys,manyshot_ys = np.array(suppopen_ys),np.array(manyshot_ys)
        cls_sampled, cls_open_ids = np.array(cls_sampled), np.array(cls_open_ids)
        
        if self.partition == 'train':
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, manyshot_xs, manyshot_ys, cls_sampled, cls_open_ids, np.array(base_ids)
        else:
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, manyshot_xs, manyshot_ys, cls_sampled, cls_open_ids

    def __len__(self):
        return self.n_episodes
