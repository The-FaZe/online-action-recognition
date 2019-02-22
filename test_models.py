import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from UCF_Dataset import TSNDataset
from Modified_CNN import TSN_model
from transforms import *
from basic_ops import ConsensusModule

import os
os.environ['TORCH_MODEL_ZOO'] = "/home/alex039u2/data/" #inzializing torch main directory

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--classInd_file', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
else:
    raise ValueError('Unkown dataset: ' + args.dataset)

#later, it will define number of segments=25 for each video, so
#number of segments is set to 1 here to take 25 snippets from each video.
model = TSN_model(num_class, 1, args.modality, base_model_name=args.arch,
                  consensus_type=args.crop_fusion_type, dropout=args.dropout)

#load the weights from the file saved during training process.
#args.weights is simply a string refers to the path of the file.
checkpoint = torch.load(args.weights)
print("epoch {}, best acc1@: {}" .format(checkpoint['epoch'], checkpoint['best_acc1']))

#base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
model.load_state_dict(checkpoint['state_dict'].items())

#specific data augmentation technique mentioned in the paper.
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
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))
    
data_loader = torch.utils.data.DataLoader(
        TSNDataset(args.test_list, num_segments=args.test_segments, 
                   new_length=1 if args.modality=='RGB' else 5, 
                   modality=args.modality, image_prefix='frame{:06d}.jpg', 
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(model.input_mean, model.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    
#this condition is to make sure that number of workers should equal
#the total number of GPUs (more search should be done on this)
if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))
    
model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)
#evaluation mode (no need for backpropagation)
model.eval()

total_num = len(data_loader.dataset)
output = []

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

def IdxtoClass(ClassIndDir):
  action_label={}
  with open(ClassIndDir) as f:
      content = f.readlines()
      content = [x.strip('\r\n') for x in content]
  f.close()

  for line in content:
      label,action = line.split(' ')
      if action not in action_label.keys():
          action_label[label]=action
          
  return action_label

def eval_video(video_data):
    """
    Evaluate single video
    video_data : Tuple has 3 elments (data in shape (crop_number,num_segments*length,H,W), label)
    return     : predictions and labels
    """
    data, label = video_data
    num_crop = args.test_crops
    #length = new_length*3
    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'RGBDiff':
        length = 15
    else:
        raise ValueError("Unknown modality "+args.modality)
    
    with torch.no_grad():
      #reshape data to be in shape of (num_segments*crop_number,length,H,W)
      input = data.view(-1, length, data.size(2), data.size(3))
      #Forword Prop
      output = model(input)
      #Covenrt output tensor to numpy array in shape (num_segments*crop_number,num_class)
      output_np = output.data.cpu().numpy().copy()
      #Reshape numpy array to (num_crop,num_segments,num_classes)
      output_np = output_np.reshape((num_crop, args.test_segments, num_class))
      #Take mean of cropped images to be in shape (num_segments,1,num_classes)
      output_np = output_np.mean(axis=0).reshape((args.test_segments,1,num_class))
    return output_np, label[0]

Label = IdxtoClass(args.classInd_file)

#i = 0 --> number of videos, data is x, and label is y
for i, (data, label) in enumerate(data_loader):
    #if we reached the end of the videos or args.max_num, exit the loop
    if i >= max_num:
        break
    #result contains our predictions and labels
    result = eval_video((data, label))
    output.append(result)
    count_time = time.time() - proc_start_time
    indxes = np.flip(np.argsort(result[0].mean(axis=0)),axis = 1)[ 0 , : 5]
    topscores = np.flip(np.sort(result[0].mean(axis=0)),axis = 1)[ 0 , : 5]
    print('total {}/{} - averageTime {:.3f} sec/video - Top5 scores: {} - Top5 actions: {} - True Labal: {}'.format(i+1,
                                                                                                                       total_num,float(count_time) / (i+1),
                                                                                                                       topscores,indxes,result[1]))


#this outputs the indices of the classified actions which can be missclassified
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
#this outputs the ground truth (right actions)
video_labels = [x[1] for x in output]

#compute accuracy using confusion matrix
cf = confusion_matrix(video_labels, video_pred).astype(float)

class_count = cf.sum(axis=1)

class_right = np.diag(cf)

class_acc = class_right / class_count

print(class_acc)
print('Accuracy {:.02f}%'.format(np.mean(class_acc)*100))

if args.save_scores is not None:
    #name_list contains directories of ordered videos
    name_list = [x.strip().split()[0] for x in open(args.test_list)]
    #order_dict is a dictionary. Keys are directories, values are ascendant numbers
    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    
    #The saved file will have scores and labels. scores has a list of tuples equal to the number of videos.
    #Each tuple is a 2-element tuple,1st element is an array of shape (num of segments,1,num of classes) which indicates the output of CNN.
    #labels is a list of the ground truth of each video (the right action).
    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]
    
    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)





