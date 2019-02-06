import argparse #used for command line interfacing.
import torch
import os
import torchvision
import torch.nn.parallel #to use multiple GPUs.
import time
import shutil
import torch.optim
import torch.backends.cudnn as cudnn 
from torch.nn.utils import clip_grad_norm_

from UCF_Dataset import TSNDataset
from Modified_CNN import TSN_model
from transforms import *
from parser_commands import parser

import subprocess 

#specify best_acc1 as a variable to deteremine the best accuracy achieved during training.
best_acc1 = 0

def main():
  global args, best_acc1
  args = parser.parse_args()
  
  if args.dataset  == 'ucf101':
    num_classes = 101
  else:
    raise ValueError('Unknown dataset: ' + args.dataset)
    
  model = TSN_model(num_classes, args.num_segments, args.modality, base_model_name=args.arch,
                   consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)
  
  crop_size = model.crop_size
  scale_size = model.scale_size
  input_mean = model.input_mean 
  input_std = model.input_std    
  policies = model.get_optim_policies() 
  train_augmentation = model.get_augmentation() 
  
  #to use multiple GPUs, args.gpus is a list (e.g. to use 4 GPUs, device_ids=[0,1,2,3]).
  model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
  
  #args.resume is an empty string to provide the path to the latest checkpoint.
  if args.resume:
    #if there is a file, do the following:
    if os.path.isfile(args.resume):
      print(("Loading checkpoint '{}'".format(args.resume)))
      #load the parameters.
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      model.load_state_dict(checkpoint['state_dict'])
      print(("Loaded checkpoint '{}' epoch {}".format(args.evaluate, checkpoint['epoch'])))
      
    else:
      print(("No checkpoint found at '{}'".format(args.resume)))
      
  #this flag allows you to enable the inbuilt cudnn auto-tuner to find 
  #the best algorithm to use for your hardware.
  #but if the input sizes change at each iteration, this will lead to worse runtime.
  cudnn.benchmark = True
  
  #different techniques of normalization is used for RGB & RGB Difference. ########
  if args.modality != 'RGBDiff':
      normalize = GroupNormalize(input_mean, input_std)
  else:
      normalize = IdentityTransform()
  
  #for RGB, we only take 1 frame per segment.
  #for RGBDiff, we take 5 consecutive frames per segment.
  if args.modality == 'RGB':
    data_length = 1
  else:
    data_length = 5
  
  #load the data using built-in PyTorch function torch.utils.data.DataLoader. ##########
  train_loader = torch.utils.data.DataLoader(
      TSNDataset(args.train_list, num_segments=args.num_segments, 
                new_length=data_length, modality=args.modality,
                image_prefix='frame{:06d}.jpg',
                transform=torchvision.transforms.Compose([
                    train_augmentation,
                    Stack(roll=args.arch=='BNInception'), #########
                    #convert RGB image with (H x W x C) to tensor of shape (C x H x W).
                    #from range [0, 255] to [0 1]
                    ToTorchFormatTensor(div=args.arch!='BNInception'),
                    normalize,
                ])),
    
      # how many subprocesses to use for data loading. 0 means that the data will be loaded 
      #in the main process.
      # Having more workers will increase the memory usage.WARNING alot of workers
      #with larg batch size will cosume all the ram.
      #Optimal value of workers is the number of cpu cores as each core is responsable
      #to deliver one of batches.
      #4 or 8 would be ok. more will distract and consume the cpu.
      batch_size=args.batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
      TSNDataset(args.val_list, num_segments=args.num_segments,
                 new_length=data_length,
                 modality=args.modality,
                 image_prefix='frame{:06d}.jpg',
                 train_val_switch=False,
                 transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                 ])),
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)
      
  #only one loss type if defined (CrossEntropy).
  if args.loss_type =='nll':
    criterion=torch.nn.CrossEntropyLoss().cuda()
  else:
    raise ValueError("Unkown loss type")
    
  print('---------------------------------------------------')  
  for group in policies:
    
    print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
      group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

  print('---------------------------------------------------')  

  #Stochastic Gradient Decent.
  optimizer = torch.optim.SGD(policies, args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)

  if args.evaluate: 
    validate(val_loader, model, criterion, 0)
    #this is used for the same reason as break in loops.
    return

  for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch, args.lr_steps)

    #train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)
    
    subprocess.run(["nvidia-smi"])
	
    #evaluate on validation set
    if (epoch+1) % args.eval_freq == 0 or epoch == args.epochs - 1:
      acc1 = validate(val_loader, model, criterion, (epoch+1) * len(train_loader))

      # remember best acc@1 and save checkpoint.
      is_best = acc1 > best_acc1
      best_acc1 = max(acc1, best_acc1)
      save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': model.state_dict(),
          'best_acc1': best_acc1,
      }, is_best)

      
def train(train_loader, model, criterion, optimizer, epoch):
    """
    train the model by doing forward and backward propagation
    inputs:
        train_loader:  single- or multi-process iterators over the dataset.
        model: TSN Model
        criterion: The loss function we use - "CrossEntropyLoss"
        optimizer: type of optimizer - "stochastic gradient descent"
        epoch: num. of epocs
    """
    
    # make instances for batch_time, data_time, losses, top1, top5 from AverageMeter class
    # to calculate the average for each of them
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # check the state of partial batch normalization
    if args.no_partialbn: 
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    
    # start to calculate the training time in seconds
    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time     
        data_time.update(time.time() - end)
        
        # Since our dataloader uses pinned memory, we can load asynchronously. 
        # Setting async to true avoids cuda synchronization. 
        # So it's mainly for the slight speed gain.
        target = target.cuda()    ##########
        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        #subprocess.run(["nvidia-smi"])
        # compute output
        output = model(input)
        #subprocess.run(["nvidia-smi"])
    
        loss = criterion(output, target) # criterion is the crossEntropyLoss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1,5))
        # update is a function in AverageMeter() class which compute the average #########
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))


        # compute gradient and do SGD step
        # "zero_grad()" this function Clears the gradients of all optimized
        # It’s important to call this before loss.backward(), 
        # otherwise you’ll accumulate the gradients from multiple passes.
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None: 
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            #if total_norm > args.clip_gradient:
                #print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
				
        optimizer.step()  

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))      
 


def validate (val_loader,model,ceriterion,logger=None):
    """
    This function for validation process
    """
    #Bunch of accumulating-based meters to calculate averages for various values
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    #notify all your layers that you are in eval mode, that way, batchnorm or
    #dropout layers will work in eval model instead of training mode.
    model.eval()                                 

    #in validation process, we don't need to backpropagate.
    with torch.no_grad():
        #Batch processing start time
        end = time.time()
        #Loop through every single batch
        #i is integer , Input is the input tensor ,Target is the truth table
        for  i, (Input,Target) in enumerate(val_loader):   
            Target = Target.cuda()
            #Input = torch.autograd.Variable(Input,volatile = True)
            #Target = torch.autograd.Variable(Target,volatile = True)       
            #Forward Propagation
            output = model(Input)
            #Calculating loss 
            loss = ceriterion(output,Target)
            #Calculateing Accuracy 
            #acc1 checks if real class == the highest output class 
            #acc5 checks if real class is in the highest 5 outputs 
            acc1,acc5 = accuracy(output, Target, topk=(1,5))
            #Update values and calculating averages
            losses.update(loss.item(), Input.size(0))
            top1.update(acc1.item(), Input.size(0))
            top5.update(acc5.item(), Input.size(0))   
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
                
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses))
        
    return top1.avg
  
  
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
  """
  save checkpoints.
  inputs:
      state: dictionary - which contain info to be saved
      is_best: boolean 
      filename: string - path to a file to save checkpoints parameters
  """
  filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
  torch.save(state, filename)
  if is_best:
      best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
      shutil.copyfile(filename, best_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all variables to 0 """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ calculate the average of the val
        inputs:
            val: integer or float num.
            n: integer
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every number of epochs (Default=30)"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

# INPUTS: output have shape of [batch_size, category_count]
# and target in the shape of [batch_size] * there is only one true class for each sample
# topk is tuple of classes to be included in the precision
# topk have to a tuple so if you are giving one number, do not forget the comma

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    """Computes the accuracy over the k top predictions for the specified values of k"""
    #we do not need gradient calculation for those
    
    #we will use biggest k, and calculate all precisions from 0 to k
    maxk = max(topk)
    batch_size = target.size(0)
    
	#topk gives biggest maxk values on dimth dimension from output
    #output was [batch_size, category_count], dim=1 so we will select biggest category scores for each batch
    # input=maxk, so we will select maxk number of classes 
    #so result will be [batch_size,maxk]
    #topk returns a tuple (values, indexes) of results
    # we only need indexes(pred)
    _, pred = output.topk(maxk, 1, True, True)  # Returns the k largest elements of the given input tensor along a given dimension.
    # then we transpose pred to be in shape of [maxk, batch_size]
    pred = pred.t()

    #we flatten target and then expand target to be like pred 
	# target [batch_size] becomes [1,batch_size]
	# target [1,batch_size] expands to be [maxk, batch_size] by repeating same correct class answer maxk times. 
	# when you compare pred (indexes) with expanded target, you get 'correct' matrix in the shape of  [maxk, batch_size] filled with 1 and 0 for correct and wrong class assignments
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))   
    res = [] 
    # then we look for each k summing 1s in the correct matrix for first k element.

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res # it should be a list with 2 elements


if __name__ == '__main__':
    os.environ['TORCH_MODEL_ZOO'] = "/home/alex039u2/data/"
    main()
