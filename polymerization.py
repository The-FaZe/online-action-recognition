import os
import time
import torch
import torch.nn.parallel
import torchvision
import torchvision.transforms as Transforms


#from matplotlib import pyplot as plt
from Modified_CNN import TSN_model
from transforms import *
from var_evaluation import Evaluation

import argparse

#--------------------Communication import--------------------------
from comms_modules.Network import set_server
from comms_modules.Streaming import rcv_frames_thread,send_results_thread
from comms_modules.TopN import Top_N

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
#parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', nargs='+', type=str,
                    help='1st and 2nd index is RGB and RGBDiff weights respectively')
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=8 ,help='Window size')
parser.add_argument('--delta', type=int, default=2,help='Value of the shift')
parser.add_argument('--psi', type=float, default=3.5)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--classInd_file', type=str, default='')
parser.add_argument('--video', type=str, default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--score_weights', nargs='+', type=float, default=[1,1.5])


parser.add_argument('--test',dest = 'test',action='store_true',help='coloring the output scores with red color in case of NoActivity case')
parser.add_argument('--p',dest= 'port' , type = int , default = 6666)
parser.add_argument('--h',dest= 'hostname' , type = str , default = 'login01')
parser.add_argument('--u',dest= 'username' , type = str , default = 'alex039u2')

args = parser.parse_args()

pre_scoresRGB = torch.zeros((args.num_segments - args.delta,101)).cuda()
pre_scoresRGBDiff = torch.zeros((args.num_segments - args.delta,101)).cuda()

#this function takes one video at a time and outputs the first 5 scores
def First_step():
  #num_crop = args.test_crops  
  test_segments = args.test_segments
  window_size = args.window_size
  
  #this function do forward propagation and returns scores
  def eval_video(data, model):
      """
      Evaluate single video
      video_data : Tuple has 3 elments (data in shape (crop_number,num_segments*length,H,W), label)
      return     : predictions and labels
      """          
      global pre_scoresRGB
      global pre_scoresRGBDiff
      
      with torch.no_grad():
          #reshape data to be in shape of (num_segments*crop_number,length,H,W)
          #Forword Propagation
          if model == 'RGB':
              input = data.view(-1, 3, data.size(1), data.size(2))
              output = torch.cat((pre_scoresRGB, model_RGB(input)))
              pre_scoresRGB = output.data[-(args.num_segments - args.delta):,]

          elif model == 'RGBDiff':
              input = data.view(-1, 18, data.size(1), data.size(2))
              output = torch.cat((pre_scoresRGBDiff, model_RGBDiff(input)))
              pre_scoresRGBDiff = output.data[-(args.num_segments - args.delta):,]
      
          #output_np = output.data.cpu().numpy().copy()    
          #Reshape numpy array to (num_crop,num_segments,num_classes)
          #output_np = output_np.reshape((num_crop, test_segments*2, num_class))
          #Take mean of cropped images to be in shape (num_segments,1,num_classes)
          #output_np = output_np.mean(axis=0).reshape((test_segments*2,1,num_class))
          #output_np = output_np.mean(axis=0)
          output_tensor = output.data.mean(dim = 0)
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
  
  cropping = torchvision.transforms.Compose([
          GroupScale(model_RGB.scale_size),
          GroupCenterCrop(model_RGB.input_size),
      ])
      
  #Required transformations
  transform = torchvision.transforms.Compose([
           cropping,
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

  softmax = torch.nn.Softmax()
  scores = torch.tensor(np.zeros((1,101)), dtype=torch.float32).cuda()
   
  frames = []  

  conn,transport = set_server(ip="0.0.0.0",port=args.port,Tunnel=True,n_conn=2,hostname= args.hostname,username=args.username)
  if conn is None:
      return 
  
  try: 
    top5_actions = Top_N(args.classInd_file)
    rcv_frames = rcv_frames_thread(connection=conn[0])
    send_results = send_results_thread(connection=conn[1],test=args.test)

    
    while (rcv_frames.isAlive() and send_results.isAlive()):
        if rcv_frames.CheckReset():
            frames = []
            rcv_frames.ConfirmReset()

        frame,status = rcv_frames.get()


        if frame is 0:
            break

        if frame is None:
           continue
      
        frame = Image.fromarray(frame)
        
        frames.append(frame)
      
        if len(frames) == args.delta*6:       
            frames = transform(frames).cuda()
            scores_RGB = eval_video(frames[0:len(frames):6], 'RGB')
            scores_RGBDiff = eval_video(frames[:], 'RGBDiff')
         
            final_scores = args.score_weights[0]*scores_RGB + args.score_weights[1] * scores_RGBDiff
            #final_scores = softmax(final_scores)
            final_scores = final_scores.data.cpu().numpy().copy()
            top5_actions.import_scores(final_scores)
            indices_TopN,_,scores_TopN = top5_actions.get_top_N_actions()
            action_checker = Evaluation(scores_TopN, args.psi)
            
            send_results.put(status=(),scores=(*indices_TopN,*scores_TopN),Actf=action_checker)
            frames = [] 
          
        else:
            send_results.put(status=(),Actf=action_checker)
          
  except (KeyboardInterrupt,IOError,OSError):
    pass
  finally:
    rcv_frames.close()
    send_results.close()
    conn[0].close()
    conn[1].close()
    if bool(transport):
        transport.close()
    

  
if __name__ == '__main__':
    os.environ['TORCH_MODEL_ZOO'] = "/home/alex039u2/data/" #adding working directory of pytorch
    First_step()
