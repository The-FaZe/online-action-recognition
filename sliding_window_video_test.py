import os 

#change this to your working directory
current_dir = r'/home/abdullah/Documents/graduation_project/codes/'
os.chdir(current_dir + 'real-time-action-recognition')

print("Current working Dorectory: ", os.getcwd())

import time
import torch
import torch.nn.parallel
import torchvision
import cv2
import torchvision.transforms as Transforms
from numpy.random import randint
import operator
import numpy as np

#from matplotlib import pyplot as plt

from Modified_CNN import TSN_model
from transforms import *

import argparse

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--classInd_file', type=str, default='')
parser.add_argument('--video', type=str, default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()



"""
--------------------Sliding Window ------------------------------------
"""
def sliding_window_aggregation_func(score, spans=[1, 2, 4, 8, 16], overlap=0.2, norm=True, fps=1):
    """
    This is the aggregation function used for ActivityNet Challenge 2016
    :param score:
    :param spans:
    :param overlap:
    :param norm:
    :param fps:
    :return:
    """
    def softmax(scores_row_vector):
        """
        the function takes the softmax of row numpy vector
        input:
            scores_row_vector: a row vector of of type numpy float32
        resturn:
            softmax_scores: row vector of of type numpy float32
        """
        torch_softmax = torch.nn.Softmax() #across rows
        scores_torch = torch.from_numpy(scores_row_vector).float().cuda()
        scores_torch = torch_softmax(scores_torch)
       
        return scores_torch.data.cpu().numpy()

    
    frm_max = score.max(axis=1)
    slide_score = []

    def top_k_pool(scores, k):
        return np.sort(scores, axis=0)[-k:, :].mean(axis=0)

    for t_span in spans:
        span = t_span * fps
        step = int(np.ceil(span * (1-overlap)))
        local_agg = [frm_max[i: i+span].max(axis=0) for i in range(0, frm_max.shape[0], step)]
        k = max(15, len(local_agg)/4)
        slide_score.append(top_k_pool(np.array(local_agg), k))

    out_score = np.mean(slide_score, axis=0)

    if norm:
        return softmax(out_score)
    else:
        return out_score

"""
----------------------------------------------------------------------------
"""

"""
-----------------------------print_topN_action class---------------
"""
class Top_N(object):
    
    client_flag = True #if import_indecies_top_N_scores it will print using the actual 
                       #scores from the server
                        #if import_scores were called it will print its tops cores
                        #(slient scores)
    def __init__(self, classInd_textfile, N=5):
        """
        input:
            classInd_textfile: the text file that contains actions
            N: is an integer refrering to top N actions
        """

        self.N = N
        self.actions_list = self.load_actions_label(classInd_textfile)
        self.scores = None
        self.top_N_scores = None
        self.top_N_actions = None
        self.indecies = None
        
        
    def load_actions_label(self, classInd):
        """
        returns a list of the actions ordered by the list index
        input:
            classInd: of type string refering the input textfile
            output:
                action_label: of type list
        """
        action_label_list=[]
        with open(classInd) as f:
            content = f.readlines()
      
        f.close()
        for line in content:
            action_label_list.append(line.split(' ')[-1].rstrip()) #splitting the number from the action
                                                            #and removing the '/n' using 
                                                            #rstrip() method
    
          
        return action_label_list    
    
    
    def import_scores(self, scores):
        """
        the method impoetes the socres of the actions
        """
        Top_N.client_flag = False
        self.scores = scores
    
    def get_top_N_actions(self):
        """
        input:
            scores: of type numpay 1d array (row vector) that contains actions'
                    scores
        returns a tuple (top N actions' indicies of type numpy array,
                                                     list of type N action,
                                                     numpy array of top N cores)
        return False if the scores were not given
        
        """
        #Handelling no scores' input
        try:
            if self.scores == None:
                return False
        except:
            pass
                
        sorted_indcies = np.argsort(self.scores)[::-1] #soring the indicies from 
                                                       #the biggesr to the loewst
                                                       
        
        sorted_indcies = sorted_indcies[:self.N]     #taking top N actioms
        
        top_N_actions = []
        top_N_scores  = []
        for i in sorted_indcies:
            top_N_actions.append(self.actions_list[i])
            top_N_scores.append(self.scores[i])
        
        self.top_N_scores = top_N_scores
        self.top_N_actions = top_N_actions
        self.indecies = sorted_indcies
        
            
        return sorted_indcies, top_N_actions, top_N_scores
    
    
    def import_indecies_top_N_scores(self, tuple_input):
        """
        the method takes a input tuple (indecies, scores)
        returns the list of actions' string
        """
        Top_N.client_flag = True
        self.indecies, self.top_N_scores = tuple_input
        
    
    def index_to_actionString(self):
        
        """
        the method takes a input tuple (indecies, scores)
        returns the list of actions' string
        """
        #Handelling no scores' input
        try:
            if self.indecies == None:
                return False
        except:
            pass
        
        top_N_actions = []
        for i in self.indecies:
            top_N_actions.append(self.actions_list[i])
            
        self.top_N_actions = top_N_actions
        
        return top_N_actions
            
        
    def __str__(self):
        
        """
        this method will be activated when doing print(object)
        """
        
        open_statement = "Top " + str(self.N) + " Actions.\n"
        
        #Handelling no scores' input
        try:
            if self.scores == None and self.indecies == None:
                return open_statement + "\nThere is no scores were given."
        except:
            pass
        
        try:
            if self.top_scores == None:
                return open_statement + "\nThere is no scores were given."
        except:
            pass
            
        if Top_N.client_flag: #if import_indecies_top_N_scores is called
            self.index_to_actionString()
        else: #if import_scores were called
            self.get_top_N_actions()
            
        action_satement = ''
        
        
        
        for i in range(self.N):
            action_satement += self.top_N_actions[i] + " : " \
                            + "{0:.4f}".format(self.top_N_scores[i]*100) + '\n'
                            
        return open_statement + action_satement
            
        
        
            



#this function takes one video at a time and outputs the first 5 scores
def one_video():

  #this function do forward propagation and returns scores
  def eval_video(data):
      """
      Evaluate single video
      video_data : Tuple has 3 elments (data in shape (crop_number,num_segments*length,H,W), label)
      return     : predictions and labels
      """
      if args.modality == 'RGB':
          length = 3
      elif args.modality == 'RGBDiff':
          length = 18
      else:
          raise ValueError("Unknown modality " + args.modality)
    
      with torch.no_grad():
          #reshape data to be in shape of (num_segments*crop_number,length,H,W)
          input = data.view(-1, length, data.size(1), data.size(2))
          #Forword Propagation
          output = model(input)
          #Reshape numpy array to (num_crop,num_segments,num_classes)
          output_torch = output.view((num_crop, test_segments, num_class))
         
          #Take mean of cropped images to be in shape (num_segments,1,num_classes)
          output_torch = output_torch.mean(dim=0).view((test_segments,1,num_class))
#          
      return output_torch     
  
  #this function used to pick 25 frames only from the whole video
  def frames_indices(frames):
      FPSeg = len(frames) // test_segments
      offset = [x*FPSeg for x in range(test_segments)]
      random_indices = list(randint(FPSeg,size=test_segments))
      frame_indices = [sum(i) for i in zip(random_indices,offset)]
      return frame_indices        
    
  num_crop = args.test_crops  
  test_segments = args.test_segments
  

  """
  --------------------Inzializations---------------------------
  """
    
  top5_actions = Top_N(args.classInd_file, 5)

  if args.dataset == 'ucf101':
      num_class = 101
  else:
      raise ValueError('Unkown dataset: ' + args.dataset)
  
  model = TSN_model(num_class, 1, args.modality,
                    base_model_name=args.arch, consensus_type='avg', dropout=args.dropout)
  
  #load the weights of your model training
  checkpoint = torch.load(args.weights)
  print("epoch {}, best acc1@: {}" .format(checkpoint['epoch'], checkpoint['best_acc1']))

  base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
  model.load_state_dict(base_dict)
  
  #test_crops is set to 1 for fast video evaluation
  if args.test_crops == 1:
      cropping = torchvision.transforms.Compose([
          GroupScale(model.scale_size),
          GroupCenterCrop(model.input_size),
      ])
  elif args.test_crops == 10:
      cropping = torchvision.transforms.Compose([
          GroupOverSample(model.input_size, model.scale_size)
      ])
  else:
      raise ValueError("Only 1 and 10 crops are supported while we got {}".format(test_crops))
      
  #Required transformations
  transform = torchvision.transforms.Compose([
           cropping,
           Stack(roll=args.arch == 'BNInception'),
           ToTorchFormatTensor(div=args.arch != 'BNInception'),
           GroupNormalize(model.input_mean, model.input_std),
                   ])
    
    
  if args.gpus is not None:
      devices = [args.gpus[i] for i in range(args.workers)]
  else:
      devices = list(range(args.workers))
    
  model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)
         
  model.eval()    

  softmax = torch.nn.Softmax() #across rows
   
  
  """
  --------------Captureing the frames from  the video-------------------
  """
  
  frames = []  
  capture = cv2.VideoCapture(args.video)
  frame_count = 0
  
  while True:
      ret, orig_frame = capture.read()
     
      if ret is True:
          frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
      else:
          break
      
      #RGB_frame is used for plotting a frame with text of top-5 scores on it
      RGB_frame = frame    
        
      #use .fromarray function to be able to apply different data augmentations
      frame = Image.fromarray(frame)
      
      frame_count += 1
      frames.append(frame) 
  
  print(frame_count)
  
  # When everything done, release the capture
  capture.release()
  #cv2.destroyAllWindows()    
  
  #to evaluate the processing time
  start_time = time.time()
  
  """
  ----------------------------------------------------------------------
  """
  '''
  images = [cv2.imread(file) for file in glob.glob(args.video + "/*.jpg")]
  
  for frame in images:    
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = Image.fromarray(frame)
      frames.append(frame)   
  '''

  
  indices = frames_indices(frames)
  num_segments = len(indices)
  
  frames_a = operator.itemgetter(*indices)(frames)
  frames_a = transform(frames_a).cuda()
  
  scores = torch.zeros((num_segments, 1, 101), dtype=torch.float32).cuda()
  scores = eval_video(frames_a)

  scores = scores.data.cpu().numpy().copy() #now we got the scores of each segment
  
  

  
  
  out_scores = np.zeros((num_segments, 1, 101), dtype=float)
  out_scores = sliding_window_aggregation_func(scores, spans=[1, 4, 8, 16], norm=True, fps=1)
  
  
  
  
  end_time = time.time() - start_time
  print("time taken: ", end_time)
  
  top5_actions.import_scores(out_scores) #importing scores to the class
  print(top5_actions)



  
if __name__ == '__main__':
    one_video()
