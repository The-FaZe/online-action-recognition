import os 
import time
#change this to your working directory
current_dir = r'/content/'
os.chdir(current_dir + 'real-time-action-recognition')

import torch
import torch.nn.parallel
import torchvision
import cv2
import torchvision.transforms as Transforms
import numpy as np
import matplotlib.pyplot as plt

from Modified_CNN import TSN_model
from transforms import *
from var_evaluation import Evaluation

import argparse

parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])

parser.add_argument('weights', nargs='+', type=str,
                    help='1st and 2nd index is RGB and RGBDiff weights respectively')
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=8)#?
parser.add_argument('--sampling_freq', type=int, default=12)#?
parser.add_argument('--delta', type=int, default=2)#?
parser.add_argument('--psi', type=float, default=3.5)#?
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--classInd_file', type=str, default='')
parser.add_argument('--video', type=str, default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--score_weights', nargs='+', type=float, default=[1,1.5])
parser.add_argument('--quality',type=str,default = '480p')

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

def add_status(frame_,s=(),x=5,y=12,font = cv2.FONT_HERSHEY_SIMPLEX
    ,fontScale = 0.4,fontcolor=(255,255,255),thickness=3,box_flag=True
    ,alpha = 0.4,boxcolor=(129, 129, 129),x_mode=None):
    
    if x_mode is 'center':
        x=frame_.shape[1]//2
    elif x_mode is 'left':
        x=frame_.shape[1]

    origin=np.array([x,y])
    y_c = add_box(frame=frame_ ,text=s ,origin=origin, font=font, fontScale=fontScale
        ,thickness=thickness ,alpha=alpha ,enable=box_flag,color=boxcolor,x_mode=x_mode)
    cv2.putText(frame_,s, tuple(origin) 
        ,font, fontScale, fontcolor, thickness)


def add_box(frame,text,origin,font,color,fontScale=1,thickness=1,alpha=0.4,enable=True,x_mode=None):
    box_dim = cv2.getTextSize(text,font,fontScale,thickness)
    if x_mode is 'center':
        origin[:] = origin - np.array([box_dim[0][0]//2,0])
    elif x_mode is 'left':
        origin[:] = origin - np.array([box_dim[0][0]+2,0])
    pt1 = origin - np.array([0,box_dim[0][1]])
    pt2 = pt1+box_dim[0]+np.array([0,box_dim[0][1]//4+thickness])
    if enable:
        overlay = frame.copy()
        cv2.rectangle(overlay,tuple(pt1),tuple(pt2),color,-1)  # A filled rectangle

        # Following line overlays transparent rectangle over the image
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return pt2[1]-pt1[1]+1

pre_scoresRGB  = torch.zeros((args.num_segments - args.delta ,101)).cuda()
pre_scoresRGBDiff =  torch.zeros((args.num_segments - args.delta ,101)).cuda()

def eval_video(data, model):
      """
      Evaluate single video
      video_data : (data in shape (1,num_segments*length,H,W))
      return     : a tensor of (101) size representing a score for certain batch of frames
      """
      global pre_scoresRGB
      global pre_scoresRGBDiff
      with torch.no_grad():
          #reshape data to be in shape of (num_segments,length,H,W)
          #Forword Propagation
          if model == 'RGB':
              input = data.view(-1, 3, data.size(1), data.size(2))
              output = torch.cat((pre_scoresRGB,model_RGB(input)),0)
              pre_scoresRGB = output.data[-(args.num_segments - args.delta):,]
              
          elif model == 'RGBDiff':
              input = data.view(-1, 18, data.size(1), data.size(2))
              output =torch.cat((pre_scoresRGBDiff,model_RGBDiff(input)),0)
              pre_scoresRGBDiff = output.data[-(args.num_segments - args.delta):,]
              
          output_tensor = output.data.mean(dim = 0,keepdim=True)
          
      return output_tensor
  
if args.dataset == 'ucf101':
  num_class = 101
else:
  raise ValueError('Unkown dataset: ' + args.dataset)
  
model_RGB = TSN_model(num_class, 1, 'RGB',
                base_model_name=args.arch, consensus_type='avg', dropout=args.dropout)
  
model_RGBDiff = TSN_model(num_class, 1, 'RGBDiff',
                base_model_name=args.arch, consensus_type='avg', dropout=args.dropout)
  
for i in range(len(args.weights)):
  #load the weights of your model training
  checkpoint = torch.load(args.weights[i])
  print("epoch {}, best acc1@: {}" .format(checkpoint['epoch'], checkpoint['best_acc1']))

  base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
  if i==0:
      model_RGB.load_state_dict(base_dict)
  else:
      model_RGBDiff.load_state_dict(base_dict)

  

  
  #Required transformations
transform = torchvision.transforms.Compose([
       GroupScale(model_RGB.scale_size),
       GroupCenterCrop(model_RGB.input_size),
       Stack(roll=args.arch == 'BNInception'),
       ToTorchFormatTensor(div=args.arch != 'BNInception'),
       GroupNormalize(model_RGB.input_mean, model_RGB.input_std),
               ])


if args.gpus is not None:
  devices = [args.gpus[i] for i in range(args.workers)]
else:
  devices = list(range(args.workers))

model_RGB = torch.nn.DataParallel(model_RGB.cuda(devices[0]), device_ids=devices)
model_RGBDiff = torch.nn.DataParallel(model_RGBDiff.cuda(devices[0]), device_ids=devices)
 
model_RGB.eval()
model_RGBDiff.eval()    


#this function takes one video at a time and outputs the first 5 scores
def one_video():
  capture = cv2.VideoCapture(args.video)
  fps_vid = capture.get(cv2.CAP_PROP_FPS)
  width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) 
  fourcc = cv2.VideoWriter_fourcc(*"XVID")
  out = cv2.VideoWriter('output.mp4', fourcc, fps_vid, (int(width), int(height)))
    

  frames = []  
  frame_count = 0
  
  if args.quality == '1080p':
      orgHigh =(0,40) 
      orgDown = (0,1020)
      font = cv2.FONT_HERSHEY_COMPLEX_SMALL
      fontScale = 3
      thickness = 3
      boxcolor= (255,255,255)
      
  if args.quality == '480p':
      orgHigh =(0,20) 
      orgDown = (0,460)
      font = cv2.FONT_HERSHEY_COMPLEX_SMALL
      fontScale = 1.5
      thickness = 2
      boxcolor= (255,255,255)
      
  action_label = label_dic(args.classInd_file)
  
  while (capture.isOpened()):
      ret, orig_frame = capture.read()
     
      if ret is True:
          frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
      else:
          print('Not getting frames')
          break  
      
      #use .fromarray function to be able to apply different data augmentations
      frame = Image.fromarray(frame)
      
      if frame_count % args.sampling_freq < 6:
          frames.append(frame)
          print('Picked frame, ', frame_count)
      
      if len(frames) == args.delta*6:
          frames = transform(frames).cuda()
          scores_RGB = eval_video(frames[0:len(frames):6], 'RGB')[0,] 
          scores_RGBDiff = eval_video(frames[:], 'RGBDiff')[0,]           
          #Fusion
          final_scores = args.score_weights[0]*scores_RGB + args.score_weights[1] * scores_RGBDiff
          #Just like np.argsort()[::-1]
          scores_indcies = torch.flip(torch.sort(final_scores.data)[1],[0])
          #Prepare List of top scores for action checker
          TopScoresList = []
          for i in scores_indcies[:5]:
              TopScoresList.append(int(final_scores[int(i)]))
          #Check for "no action state"
          action_checker = Evaluation(TopScoresList, args.psi)
          #qaction_checker = True
          if not action_checker:
              print('No Assigned Action')
          for i in scores_indcies[:5]:
              print('%-22s %0.2f'% (action_label[str(int(i)+1)], final_scores[int(i)]))
          print('<----------------->')
          frames = []     
          

      
      add_status(orig_frame ,x=orgHigh[0] ,y=orgHigh[1] ,s='Frame: {}'.format(frame_count),
                font = font, fontScale = fontScale-.7 ,fontcolor=(0,255,255) ,thickness=thickness-1
                ,box_flag=True, alpha = .5, boxcolor=boxcolor, x_mode='center')
      
      if frame_count <= args.sampling_freq*args.delta:
          add_status(orig_frame ,x=orgDown[0] ,y=orgDown[1] ,s='Start Recognition ...',
                font = font, fontScale = fontScale ,fontcolor=(255,0,0) ,thickness=thickness
                ,box_flag=True, alpha = .5, boxcolor=boxcolor, x_mode='center')              
      else:
          
          if action_checker:
              add_status(orig_frame ,x=orgDown[0] ,y=orgDown[1] ,s='Detected Action: {}'.format(action_label[str(int(scores_indcies[0])+1)]),
               font = font, fontScale = fontScale ,fontcolor=(0,0,0) ,thickness=thickness
                ,box_flag=True, alpha = .5, boxcolor=boxcolor, x_mode='center')
                    
          else:
              add_status(orig_frame ,x=orgDown[0] ,y=orgDown[1] ,s='No Assigned Action',
               font = font, fontScale = fontScale ,fontcolor=(0,0,255) ,thickness=thickness
                ,box_flag=True, alpha = .5, boxcolor=boxcolor, x_mode='center')  
              
      frame_count += 1          
      out.write(orig_frame)
      
  
  # When everything done, release the capture
  capture.release()
  cv2.destroyAllWindows()    
  
  
if __name__ == '__main__':
    one_video()