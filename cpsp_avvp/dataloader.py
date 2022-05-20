import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import random
import pickle as pkl

categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
 
def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y



class LLP_dataset(Dataset):

    def __init__(self, label, audio_dir, video_dir, st_dir, train=None, transform=None):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform


        self.train = train

        labels_to_idx = {}
        for i in range(len(categories)):
            labels_to_idx[i] = []        

        for idx in range(len(self.filenames)):
            row = self.df.loc[idx, :]
            ids = row[-1].split(',')
            label = ids_to_multinomial(ids)

            if len(ids)==1:
                for c in range(len(categories)):
                    if label[c] == 1:
                        labels_to_idx[c].append(idx)
                
        self.labels_to_idx = labels_to_idx
    

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        v = 0.9
        pa = label
        pv = v * label + (1 - v) * 0.5

        if self.train:
            # find the label idx
            if len(ids) == 1:
                for c in range(25):
                    if label[c] == 1:
                        break
                idx2 = random.choice(self.labels_to_idx[c])
            else:
                idx2 = random.randint(0, len(self.filenames)-1)

            row = self.df.loc[idx2, :]
            name = row[0][:11]
            audio2 = np.load(os.path.join(self.audio_dir, name + '.npy'))
        else:
            audio2 = np.array(1)    

        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 'label': label, 'audio2':audio2, 'idx':np.array(idx),
                  'pa': pa, 'pv': pv}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']
            return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(label)}
        else:
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']
            pa = sample['pa']       
            pv = sample['pv']
            return {'audio': torch.from_numpy(audio), 'video_s': torch.from_numpy(video_s),
                    'video_st': torch.from_numpy(video_st),
                    'pa':torch.from_numpy(pa), 'pv':torch.from_numpy(pv),
                    'label': torch.from_numpy(label), 'audio2':torch.from_numpy(sample['audio2']), 'idx':torch.from_numpy(sample['idx'])}
