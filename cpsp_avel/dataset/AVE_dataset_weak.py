import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import pdb



class AVEDataset_VGG(Dataset):
    def __init__(self, data_root, args=None, split='train', category_num=28):
        super(AVEDataset_VGG, self).__init__()
        self.split = split
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # Now for the weakly supervised task
        self.labels_path = os.path.join(data_root, 'right_labels.h5') # original labels for testing
        self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # video-level labels
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.h5_isOpen = False


    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.clean_labels = h5py.File(self.dir_labels_path, 'r')['avadataset']

            if self.split == 'test':
                self.labels = h5py.File(self.labels_path, 'r')['avadataset']

            self.h5_isOpen = True

        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        if self.split == 'train':
            label = self.clean_labels[sample_index]
            label = label[:-1]
        else:
            # for testing
            label = self.labels[sample_index]

        return visual_feat, audio_feat, label


    def __len__(self):
        sample_order = h5py.File(self.sample_order_path, 'r')['order']
        length = len(sample_order)
        return length


class AVEDataset_ResNet(Dataset):
    def __init__(self, data_root, args, split='train', category_num=28):
    # def __init__(self, data_root, split='train'):
        super(AVEDataset_ResNet, self).__init__()
        self.split = split
        self.category_num = category_num
        self.all_anno_path = os.path.join(data_root, "Annotations.txt")
        self.split_index_file_path = os.path.join(data_root, "%sSet.txt"%self.split)
        self.vis_maps_path = args.vis_maps_path
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # Now for the weakly supervised task
        self.labels_path = os.path.join(data_root, 'right_labels.h5') # original labels for testing
        self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # video-level labels
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.clean_labels = h5py.File(self.dir_labels_path, 'r')['avadataset']
            if self.split == 'test':
                self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.h5_isOpen = True
        all_anno_data = pd.read_csv(self.all_anno_path, sep='&')
        all_video_id_list = all_anno_data['VideoID'].tolist()

        with open(self.split_index_file_path, "r") as fr:
            video_list = fr.readlines()
        video_name = video_list[index].split('&')[1]
        label_index = all_video_id_list.index(video_name)

        if self.split == 'train':
            label = self.clean_labels[label_index]
            label = label[:-1]
        else:
            # for testing
            label = self.labels[label_index]


        vis_map_path = os.path.join(self.vis_maps_path, self.split)
        visual_feat = self._load_maps(vis_map_path, video_name) # [B, 10, 512, 7, 7]
        audio_feat = self.audio_feature[label_index] # [B, 10, 128]

        return visual_feat, audio_feat, label


    def _load_maps(self, vis_map_path, video_name):
        with open(os.path.join(vis_map_path, "%s.pkl"%video_name), 'rb') as fr:
            vis_fea_map = pickle.load(fr)
        vis_fea_map = torch.from_numpy(vis_fea_map)
        return vis_fea_map
      

    def __len__(self):
        with open(self.split_index_file_path, "r") as fr:
            video_list = fr.readlines()
        return len(video_list)



if __name__ == "__main__":
    '''Dataset'''
    train_dataloader = torch.utils.data.DataLoader(
        AVEDataset_VGG('../../ave_data/', args=None, split='train'),
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    for n_iter, batch_data in enumerate(train_dataloader):
        visual_feature, audio_feature, labels = batch_data
        pdb.set_trace()
    pdb.set_trace()
    
