import time
import torch
import torch.nn.parallel
import torchvision
import torchvision.transforms as Transforms


#from matplotlib import pyplot as plt

from Modified_CNN import TSN_model
from transforms import *

import argparse

#--------------------Communication import--------------------------
from communication import Streaming, TopN, Network
import threading
import multiprocessing as mp

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
def First_step():
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
  frame_count = 0 
  try: 
    top5_actions = Top_N(classInd_file)
    Tunnel_ = True
    conn,T_thr = Network.set_server(port=6666,Tunnel=Tunnel_,n=1)
    rcv_frames = Streaming.rcv_frames_thread(connection=conn[0])
    send_results = Streaming.send_results_thread(connection=conn[1],scores_f=False)
    while (rcv_frames.isAlive() and send_results.isAlive()):
        frame,status = rcv_frames.get()
        if frame is 0:
          break
        frame_count += 1
        frame = Image.fromarray(frame)
        
        if args.modality == 'RGB':             
            frames.append(frame)
            if frame_count % 5 == 0 and frame_count != 0:
                frames = transform(frames).cuda()
                scores = eval_video(frames)
                scores = softmax(torch.FloatTensor(scores))
                scores = scores.data.cpu().numpy().copy()
                top5_actions.import_scores(scores)
                indecies,_,scores = top5_actions.get_top_N_actions()
                send_results.put(status=status,scores=(*indecies,*scores))
                
               frames = []
            else:
                send_results.put(status=status)
  except (KeyboardInterrupt,IOError,OSError) as e:
    pass
  finally:
    rcv_frames.close()
    send_results.close()
    conn[0].close()
    conn[1].close()
    

  
if __name__ == '__main__':
    First_step()
