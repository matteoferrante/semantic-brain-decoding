import h5py

import mat73
import hdf5storage
import bdpy

from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp
import pickle
from scipy.io import loadmat
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn import preprocessing
import torch
from torch import nn
from PIL import Image

class data_handler():
    """Generate batches for FMRI prediction
    frames_back - how many video frames to take before FMRI frame
    frames_forward - how many video frames to take after FMRI frame
    
    
    -> NOT USEFUL FOR VOXEL
    
    """

    def __init__(self, h5_file ,test_img_csv = 'KamitaniData/imageID_test.csv',train_img_csv = 'KamitaniData/imageID_training.csv',voxel_spacing =3,log = 0 ):
        
        
        dat=bdpy.BData(h5_file)
        self.dat=dat
        
        self.data = dat.dataset
        #self.sample_meta = mat['dataset'][:,:3]
        #meta = mat['metadata']


        self.meta_keys = list(l for l in dat.metadata.key)
        self.meta_desc = list(l for l in dat.metadata.description )
        
        
#         self.data = mat['dataSet'][:,3:]
#         self.sample_meta = mat['dataSet'][:,:3].astype(np.object_)
#         meta = mat['metaData']
        
        

        # self.voxel_meta = np.nan_to_num(meta[0][0][2][:,3:])
        test_img_df = pd.read_csv(test_img_csv, header=None)
        train_img_df =pd.read_csv(train_img_csv, header=None)
        
        self.test_img_df=test_img_df
        self.train_img_df=train_img_df
        
        self.test_img_id = test_img_df[0].values
        self.train_img_id = train_img_df[0].values
        self.sample_type = {'train':1 , 'test':2 , 'test_imagine' : 3}
        self.voxel_spacing = voxel_spacing

        self.log = log

    def get_meta_field(self,field = 'DataType'):
        index = self.meta_keys.index(field)
        print("index",index)
        if(index <3): # 3 first keys are sample meta
            return self.sample_meta[:,index]
        else:
            return self.voxel_meta[index]


    def print_meta_desc(self):
        print(self.meta_desc)

    def get_labels(self, imag_data = 0,test_run_list = None):
        le = preprocessing.LabelEncoder()

        img_ids = self.dat.select("stimulus_id").squeeze()
        type = self.dat.select('DataType')
        train = (type == self.sample_type['train']).squeeze()
        test = (type == self.sample_type['test']).squeeze()
        imag = (type == self.sample_type['test_imagine']).squeeze()
        
        
        img_ids_train = img_ids[train]
        img_ids_test = img_ids[test]
        img_ids_imag = img_ids[imag]
        
        #print(img_ids_train)

        
        img_train_filenames=[self.train_img_df[self.train_img_df[0]==i][1].values for i in img_ids_train]
        img_test_filenames=[self.test_img_df[self.test_img_df[0]==i][1].values for i in img_ids_test]
        
        self.img_train_filenames=[i.squeeze() for i in img_train_filenames]
        self.img_test_filenames=[i.squeeze() for i in img_test_filenames]
        
        #print(img_train_filenames)

        train_labels  = []
        test_labels  =  []
        imag_labels = []
        for id in img_ids_test:
            idx = (np.abs(id - self.test_img_id)).argmin()
            test_labels.append(idx)

        for id in img_ids_train:
            idx = (np.abs(id - self.train_img_id)).argmin()
            train_labels.append(idx)

        for id in img_ids_imag:
            idx = (np.abs(id - self.test_img_id)).argmin()
            imag_labels.append(idx)

        if (test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]

            select = np.in1d(run, test_run_list)
            test_labels = test_labels[select]

        #imag_labels = le.fit_transform(img_ids_imag)
        if(imag_data):
            return np.array(train_labels), np.array(test_labels), np.array(imag_labels)
        else:
            return np.array(train_labels),np.array(test_labels)





    def get_data(self,normalize =1 ,roi = 'ROI_VC',imag_data = 0,test_run_list = None):   # normalize 0-no, 1- per run , 2- train/test seperatly
        type = self.dat.select('DataType')
        train = (type == self.sample_type['train']).squeeze()
        test = (type == self.sample_type['test']).squeeze()
        test_imag = (type == self.sample_type['test_imagine']).squeeze()
        test_all  = np.logical_or(test,test_imag)

        roi_select = self.dat.select(roi)
        #data = self.data[:,roi_select]
        
        data=roi_select #not sure!
        
        if(self.log ==1):
            data = np.log(1+np.abs(data))*np.sign(data)


        if(normalize==1):

            run = self.dat.select('Run').astype('int').squeeze()-1
            num_runs = np.max(run)+1
            data_norm = np.zeros(data.shape)

            for r in range(num_runs):
                data_norm[r==run] = sklearn.preprocessing.scale(data[r==run])
            train_data = data_norm[train]
            test_data  = data_norm[test]
            test_all = data_norm[test_all]
            test_imag = data_norm[test_imag]

        else:
            train_data = data[train]
            test_data  =  data[test]
            if(normalize==2):
                train_data = sklearn.preprocessing.scale(train_data)
                test_data = sklearn.preprocessing.scale(test_data)


        if(self.log ==2):
            train_data = np.log(1+np.abs(train_data))*np.sign(train_data)
            test_data = np.log(1+np.abs(test_data))*np.sign(test_data)
            train_data = sklearn.preprocessing.scale(train_data)
            test_data = sklearn.preprocessing.scale(test_data)



        test_labels =  self.get_labels()[1]
        imag_labels = self.get_labels(1)[2]
        num_labels = max(test_labels)+1
        test_data_avg = np.zeros([num_labels,test_data.shape[1]])
        test_imag_avg = np.zeros([num_labels,test_data.shape[1]])

        if(test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]

            select = np.in1d(run, test_run_list)
            test_data = test_data[select,:]
            test_labels = test_labels[select]

        for i in range(num_labels):
            test_data_avg[i] = np.mean(test_data[test_labels==i],axis=0)
            test_imag_avg[i] = np.mean(test_imag[imag_labels == i], axis=0)
        if(imag_data):
            return train_data, test_data, test_data_avg,test_imag,test_imag_avg

        else:
            return train_data, test_data, test_data_avg

    def get_voxel_loc(self):
        x = self.get_meta_field('voxel_x')
        y = self.get_meta_field('voxel_y')
        z = self.get_meta_field('voxel_z')
        dim = [int(x.max() -x.min()+1),int(y.max() -y.min()+1), int(z.max() -z.min()+1)]
        return [x,y,z] , dim

    
    def get_filenames(self):
        return self.img_train_filenames,self.img_test_filenames
    
    
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,fmri_data,image_paths,transform=None):
        self.fmri_data=fmri_data
        self.image_paths=image_paths
        self.transform=transform
    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        
        fmri=self.fmri_data[idx]
        
        
        image = Image.open(self.image_paths[idx]).convert("RGB")
        

        if self.transform:
            image = self.transform(image)

        return fmri,image
    
    
class ConvEncoder(nn.Module):
    def __init__(self,conv_channels=[32,64,128],bn=False,latent_dim=512,final_dim=4):
        super().__init__()
        self.latent_dim=latent_dim
        conv_layers=[]
        for c in conv_channels:
            conv_layers.append(nn.LazyConv2d(out_channels=c,kernel_size=4,stride=2,padding=1))
            if bn:
                conv_layers.append(nn.BatchNorm2d(c))
            conv_layers.append(nn.GELU())
        conv_layers.append(nn.AdaptiveAvgPool2d(final_dim))
        conv_layers.append(nn.Flatten())
        
        conv_layers.append(nn.LazyLinear(latent_dim))
        self.model=nn.Sequential(*conv_layers)
        
    
    def forward(self,x):
        return self.model(x)
            
        
class ConvDecoder(nn.Module):
    def __init__(self,conv_channels=[64,32,16],out_channels=1,bn=False,latent_dim=512,target_side=96):
        super().__init__()
        conv_layers=[]
        
        self.downsampled_side=target_side//2**len(conv_channels)
        self.predecoder=nn.LazyLinear(conv_channels[0]*self.downsampled_side**2)
        
        
        for c in conv_channels:
            conv_layers.append(nn.LazyConvTranspose2d(out_channels=c,kernel_size=4,stride=2,padding=1))
            if bn:
                conv_layers.append(nn.BatchNorm2d(c))
            conv_layers.append(nn.GELU())
        
        conv_layers.append(nn.LazyConvTranspose2d(out_channels=out_channels,kernel_size=3,stride=1,padding=1))
        conv_layers.append(nn.Sigmoid())
        self.model=nn.Sequential(*conv_layers)
    
    def forward(self,x):
        
        #reshape features
        bs=x.shape[0]
        x=self.predecoder(x)
        x=x.view(bs,-1,self.downsampled_side,self.downsampled_side)
        
        return self.model(x)        
    
    
class AE(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        
        
    def forward(self,x):
        
        z=self.encoder(x)
        x2=self.decoder(z)
        
        return x2

                 
         