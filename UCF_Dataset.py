# Sampling our video snippets to pick random frames.

import torch.utils.data as data     #PyTorch library to use our dataset class as a subclass of torch dataset.
from PIL import Image               #Library to let us interact with different images.
import os                           #library to interact with your OS whether it is Windows, Linux or MAC                                                 
import numpy as np                  #numpy library for linear algebra and matrices operations
from numpy.random import randint    #for random number generator


class Video_Info(object):
    '''
      line: Vidoes info imported form list_file [Directory -- # of frames -- label]
    '''
    def __init__(self, line):
        self.data = line
        
    @property  
    def path(self):
        return self.data[0]

    @property
    def num_frames(self):
        return int(self.data[1])

    @property
    def label(self):
        return int(self.data[2])


class TSNDataset(data.Dataset):
    '''
    file_list         : FileList.txt 
    num_segments      : Default = 3
    new_length        : The num of sequentially picked frames from 1 segment (1 'RGB' , more with 'RGBDiff')
    modality          : Takes two strings 'RGB' or 'RGBDiff'
    image_prefix      : Prefix of the frames name
    transform         : Pytorch composion of transformations (Data Augmentation)
    train_val_switch : True for train and False for validation

    The overall use of this class is :
    >>Pick = TSNdataset(......)            #Creating an object
    >>Pick[Index]                          #Index : is the index for a video in the dataset (0:13320)
    '''
    def __init__(self, file_list,num_segments=3, new_length=1, modality='RGB',
                 image_prefix='frame{:06d}.jpg', transform=None, test_mode=False, train_val_switch=True):

       
        self.file_list = file_list
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_prefix = image_prefix
        self.transform = transform
        self.test_mode = test_mode
        self.train_val_switch = train_val_switch                                      

        if modality == 'RGBDiff':
          self.new_length +=1
          
        #List of Objects, Everyone of them corresponding to a video
        self.VidInfoList = [Video_Info(x.strip().split(' '))  for x in open(self.file_list)]
    
  
    def Train_Sample_indices(self,info):
        '''
        info: Object has info about one video
        This function determine the indices for the chosen frames
        '''
        if info.num_frames < self.num_segments:
            #Return the 1st frame many times
            return np.zeros((self.num_segments,))                     
        
        #Frames per segment
        FPSeg = info.num_frames // self.num_segments      
        #The first index for every segment                                           
        offset = [x*FPSeg for x in range(self.num_segments)]   
        #Chosen frames form every segment                                     
        frame_indices = list(randint(FPSeg,size=self.num_segments))  
        #elment-wise sumtion of offset and smaple_indices
        sample_indices = [sum(i) for i in zip(frame_indices,offset)]
        return np.array(sample_indices)+1                            
        #Note: #Frames indices starts from 000001
  
    def Val_Sample_indices(self,info):
        '''
        info : Object has info about one video
        This function determine the indices for the chosen frames
        '''
        if info.num_frames < self.num_segments:
            return np.zeros((self.num_segments,))

        FPSeg = info.num_frames / float(self.num_segments)
        #Get the middle frame for every segment
        sample_indices = [int(FPSeg*( x + 1/2.0 )) for x in range(self.num_segments)]               
        return np.array(sample_indices)+1
  
  
    def Test_Sample_indices(self,info):
        '''
        info : Object has info about one video
        This function determine the indices for the chosen frames
        '''    
        if info.num_frames < self.num_segments:
            return np.zeros((self.num_segments,))

        FPSeg = info.num_frames / float(self.num_segments)
        sample_indices = [int(FPSeg*( x + 1/2.0 )) for x in range(self.num_segments)]
        return np.array(sample_indices)+1
    
    
    def Vid2Frames(self, info, indices):
        '''
        info : Object has info about one video
        indices  : Indices for the chosen frames
        '''
        images = []
        Expand_indices = []

        for idx in indices: 
            #For every frame in the chosen frames                                                                                
            for f in range(self.new_length): 
            #If new lenght is more than 1 , we'll stack num of frames from 1 seg
              Expand_indices.append(idx+f)
              if idx+f <= info.num_frames:                                                                     
                Img = [Image.open(os.path.join(info.path,self.image_prefix.format(idx+f))).convert('RGB')]                
                images.extend(Img)                                                                                         
              else:
                images.extend(Img)                                                                                        
            
            #Apply transform stuff on the chosen frames
            processed_frames = self.transform(images)                                   
        return processed_frames , info.label 
        
          
    def __getitem__(self,idx):
        '''
        idx : Index for a video into the dataset
        The main perpuse of this function is to interact 'nicer' with the object of the class
        >>TestObj[3] 
        This line output a tuple (List of chosen frames,label) for the 3rd video in the dataset
        '''
        vid_info = self.VidInfoList[idx]
        if self.test_mode:
            indices = self.Test_Sample_indices(vid_info)
        else:
            indices = self.Train_Sample_indices(vid_info) if self.train_val_switch else self.Val_Sample_indices(vid_info)
        return self.Vid2Frames(vid_info, indices)
    
    def __len__(self):
        return len(self.VidInfoList)

