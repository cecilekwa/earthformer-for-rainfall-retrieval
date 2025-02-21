import glob
import random
import numpy as np
import h5py
from datetime import datetime, timedelta
import pandas as pd
import os

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from scipy.ndimage import zoom # Upsampling of SEVIRI!

from utils.fixedValues import NORM_DICT_TOTAL as normDict


class kucukINCAdataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, 
                img_list, 
                nt_start=0, 
                nt_future=0, # Set this to 0 if you want to load the whole sequence as past data!
        ): # 
        self.nt_start = nt_start
        self.nt_future = nt_future
        if nt_future == 24:
            self.nt_past = 25
        elif nt_future == 12:
            self.nt_past = 13
        elif nt_future == 18:
            self.nt_past = 19
        elif nt_future == 0:
            self.nt_past = 1
        elif nt_future == 8:
            self.nt_past = 8
        elif nt_future == 9:
            self.nt_past = 9

        else:
           raise ValueError(f"Unexpected nt_future value = {self.nt_future}. Please check the configuration.") 
        ##
        self.img_list = img_list
        print(f'num ims: {len(img_list)}')
        
        dat_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.dem = h5py.File(dat_dir+'/data/Auxillary/DEM_resampled_filled_Ghana.h5','r')['DEM'][:,:]
        self.lat = h5py.File(dat_dir+'/data/Auxillary/grid_2d.h5','r')['lat'][:, :]
        self.lon = h5py.File(dat_dir+'/data/Auxillary/grid_2d.h5','r')['lon'][:, :]
        #
        self.dem = (self.dem - normDict['dem']['mean']) / normDict['dem']['std']
        self.lat = (self.lat - normDict['lat']['mean']) / normDict['lat']['std']
        self.lon = (self.lon - normDict['lon']['mean']) / normDict['lon']['std']

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.img_list)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.img_list[index]
        eventFile = h5py.File(file_name, 'r')
        # Get the timestamp from file_name!
        timestamp = file_name.split('_')[-1][0:11]
        # Make it a time object!
        timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M')
        # Get the day of year and hour of day!        
        time_of_day = timestamp.hour + timestamp.minute / 60 
        ## Work on seviri!
        # Read the data
        seviri = eventFile['SEVIRI'][self.nt_start:(self.nt_start+self.nt_past) + 1,:,:,:] # (t, h, w, c) 
        imerg_precip_y = eventFile['IMERG'][0,:,:]
        
        ## Time to normalize the data!
        seviri[:, :, :, 0] = (seviri[:, :, :, 0] - normDict['ch01']['mean']) / normDict['ch01']['std']
        seviri[:, :, :, 1] = (seviri[:, :, :, 1] - normDict['ch02']['mean']) / normDict['ch02']['std']
        seviri[:, :, :, 2] = (seviri[:, :, :, 2] - normDict['ch03']['mean']) / normDict['ch03']['std']
        seviri[:, :, :, 3] = (seviri[:, :, :, 3] - normDict['ch04']['mean']) / normDict['ch04']['std']
        seviri[:, :, :, 4] = (seviri[:, :, :, 4] - normDict['ch05']['mean']) / normDict['ch05']['std']
        seviri[:, :, :, 5] = (seviri[:, :, :, 5] - normDict['ch06']['mean']) / normDict['ch06']['std']
        seviri[:, :, :, 6] = (seviri[:, :, :, 6] - normDict['ch07']['mean']) / normDict['ch07']['std']
        seviri[:, :, :, 7] = (seviri[:, :, :, 7] - normDict['ch08']['mean']) / normDict['ch08']['std']
        seviri[:, :, :, 8] = (seviri[:, :, :, 8] - normDict['ch09']['mean']) / normDict['ch09']['std']
        seviri[:, :, :, 9] = (seviri[:, :, :, 9] - normDict['ch10']['mean']) / normDict['ch10']['std']
        seviri[:, :, :, 10] = (seviri[:, :, :, 10] - normDict['ch11']['mean']) / normDict['ch11']['std']

        imerg_precip_y[imerg_precip_y < 0.1] = 0.02
        
        imerg_precip_y = np.log10(imerg_precip_y)
        
        # Add DoY and HoD as sine waves!
        doy_sin = np.ones((248, 184)) * np.sin(timestamp.timetuple().tm_yday * 2 * np.pi / 365.2425)
        doy_cos = np.ones((248, 184)) * np.cos(timestamp.timetuple().tm_yday * 2 * np.pi / 365.2425)
        hod_sin = np.ones((248, 184)) * np.sin(time_of_day * 2 * np.pi / 24)
        hod_cos = np.ones((248, 184)) * np.cos(time_of_day * 2 * np.pi / 24)
        
        # Get the number of zero layers in the first 'time steps' of aux data!
        # missing_steps = self.nt_past - inca_cape.shape[0] - 1 - 2 - 2 - 2 # 1 for dem, 2 for coords, 2 for sine waves of time, 2 for cos waves of time
        missing_steps = self.nt_past - 1 - 2 - 2 - 2 # 1 for dem, 2 for coords, 2 for sine waves of time, 2 for cos waves of time
        # Combine the aux 'band'
        combined_aux = np.concatenate((np.zeros((missing_steps, 248, 184)),
                                       self.dem[np.newaxis, :, :], 
                                       self.lat[np.newaxis, :, :], self.lon[np.newaxis, :, :], 
                                       doy_sin[np.newaxis, :, :], doy_cos[np.newaxis, :, :], 
                                       hod_sin[np.newaxis, :, :], hod_cos[np.newaxis, :, :]), axis=0)
        ## Finally, concatenate all of the data!

        sample_past = np.concatenate((seviri, combined_aux[:, :, :, np.newaxis]), axis=-1)

        
        # sample_future = inca_precip_y[:, :, :, np.newaxis]
        sample_future = imerg_precip_y[np.newaxis, :, :, np.newaxis]
        ## Push them to torch and return a sample
        sample_past = torch.from_numpy(sample_past.astype(np.float32))
        sample_future = torch.from_numpy(sample_future.astype(np.float32))
        sample = {'sample_past': sample_past, 'sample_future': sample_future, 'name': file_name}
        return sample

class kucukINCAdataModule(pl.LightningDataModule):
    def __init__(self, 
                params, 
                data_dir,
                stage='train', 
                ):
        super().__init__() 
        self.params = params

        ## Let's define the fileList here!
        trainFilesTotal = glob.glob(data_dir+'train/*.hdf5') 
        valFilesTotal = glob.glob(data_dir+'val/*.hdf5')
        random.seed(0)
        random.shuffle(trainFilesTotal)  # shuffle the file list randomly

        self.testFiles = glob.glob(data_dir+'test/*.hdf5')

        self.trainFiles = trainFilesTotal
        self.valFiles = valFilesTotal

    def setup(self, stage=None):
        if self.params['out_len'] == 24:
            nt_start = 0
            nt_future = 24
        elif self.params['out_len'] == 12:
            nt_start = 11
            nt_future = 12
        elif self.params['out_len'] == 18:
            nt_start = 5
            nt_future = 18
        elif self.params['out_len'] == 1:
            nt_start = 0
            nt_future = 9
        if stage == 'train' or stage is None:
            self.train_dataset = kucukINCAdataset(img_list=self.trainFiles, 
                                                  nt_start=nt_start, nt_future=nt_future,
                                                  )
            self.val_dataset = kucukINCAdataset(img_list=self.valFiles,
                                                nt_start=nt_start, nt_future=nt_future,
                                                )
        if stage == 'test':
            self.test_dataset = kucukINCAdataset(img_list=self.testFiles, 
                                                 nt_start=nt_start, nt_future=nt_future,
                                                 )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['BATCH_SIZE'], shuffle=True, num_workers=self.params['NUM_WORKERS'])

    def val_dataloader(self):
        # No shuffling!
        return DataLoader(self.val_dataset, batch_size=self.params['BATCH_SIZE'], shuffle=False, num_workers=self.params['NUM_WORKERS'])

    def test_dataloader(self):
        # Batch size is 1 for testing! No shuffling!
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.params['NUM_WORKERS'])
    
    #### These properties are added from the original repository!
    @property
    def num_train_samples(self):
        return len(self.train_dataset)

    @property
    def num_val_samples(self):
        return len(self.val_dataset)

    @property
    def num_test_samples(self):
        return len(self.test_dataset)

