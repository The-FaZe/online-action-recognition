import os 

#change this to your working directory
current_dir = r'~/~/Documents/graduation_project/codes/real-time-action-recognition/untrimmed_work'
#os.chdir(current_dir + 'real-time-action-recognition')

import time
import torch
import torch.nn.parallel
import torchvision
import cv2
import torchvision.transforms as Transforms
from numpy.random import randint
import operator

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

#this function returns a dictionary (keys are string label numbers & values are action labels)
def label_dic(classInd):
  action_label={}
  with open(classInd) as f:
      content = f.readlines()
      content = [x.strip('\r\n') for x in content]
  f.close()

  for line in content:
      label, action = line.split(' ')
      if action not in action_label.keys():
          action_label[label] = action
          
  return action_label

#this function takes one video at a time and outputs the first 5 scores
def one_video():
  num_crop = args.test_crops  
  test_segments = args.test_segments
  
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
          output_np = output.data.cpu().numpy().copy()
          #Reshape numpy array to (num_crop,num_segments,num_classes)
          output_np = output_np.reshape((num_crop, test_segments, num_class))
          #Take mean of cropped images to be in shape (num_segments,1,num_classes)
          output_np = output_np.mean(axis=0).reshape((test_segments,1,num_class))
          output_np = output_np.mean(axis=0)
      return output_np   
    
    
  action_label = label_dic(args.classInd_file)

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

  softmax = torch.nn.Softmax()
  scores = torch.tensor(np.zeros((1,101)), dtype=torch.float32).cuda()
   
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
  
  '''
  images = [cv2.imread(file) for file in glob.glob(args.video + "/*.jpg")]
  
  for frame in images:    
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = Image.fromarray(frame)
      frames.append(frame)   
  '''
  
  #this function used to pick 25 frames only from the whole video
  def frames_indices(frames):
      FPSeg = len(frames) // test_segments
      offset = [x*FPSeg for x in range(test_segments)]
      random_indices = list(randint(FPSeg,size=test_segments))
      frame_indices = [sum(i) for i in zip(random_indices,offset)]
      return frame_indices
  
  indices = frames_indices(frames) 
  frames_a = operator.itemgetter(*indices)(frames)
  frames_a = transform(frames_a).cuda()
  
  scores = eval_video(frames_a)
  scores = softmax(torch.FloatTensor(scores))
  scores = scores.data.cpu().numpy().copy()
  
  end_time = time.time() - start_time
  print("time taken: ", end_time)
  
  # Display the resulting frame and the classified action
  font = cv2.FONT_HERSHEY_SIMPLEX
  y0, dy = 300, 40
  k=0
  print('Top 5 actions: ')
  #get the top-5 classified actions
  for i in np.argsort(scores)[0][::-1][:5]:
      print('%-22s %0.2f%%' % (action_label[str(i+1)], scores[0][i] * 100))
      #this equation is used to print different actions on a separate line
      y = y0 + k * dy
      k+=1
      cv2.putText(RGB_frame, text='{} - {:.2f}'.format(action_label[str(i+1)],scores[0][i]), 
                       org=(5,y),fontFace=font, fontScale=1,
                       color=(0,0,255), thickness=2)
  
  #save the frame in your current working directory
  cv2.imwrite(current_dir + 'text_frame'+'.png', RGB_frame)
  #plt.imshow(img)
  #plt.show()    
  
if __name__ == '__main__':
    one_video()
