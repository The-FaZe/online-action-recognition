import os 
import torch
import torchvision
import cv2
import torchvision.transforms as Transforms
from TCP import Frames_rcv

from Modified_CNN import TSN_model
from transforms import *

os.chdir(r'C:\Driver E\Jimy\Machine_Learning\Graduation_Project\webcam_interface\Trial\real-time-action-recognition')


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

def webcam(ip,port,weight_dir,test_crops,ClassIndDir,arch = 'BNInception'):
  
  #Perpare the model
  model = TSN_model(num_classes=101, num_segments=1, modality='RGB',
                  consensus_type='avg', base_model_name='BNInception',
                  new_length=None, before_softmax=True, dropout=0.8,
                  crop_num=1, partial_bn=True)
  
  model.eval()
  
  model.cuda()
  
  #Map class idx with class name into dictionary
  idx_to_class = IdxtoClass(ClassIndDir)
  
  #Load Weights
  checkpoint = torch.load(weight_dir)
  print("epoch {}, best acc1@: {}" .format(checkpoint['epoch'], checkpoint['best_acc1']))

  base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
  model.load_state_dict(base_dict)
  
  
  if test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(model.scale_size),
        GroupCenterCrop(model.input_size),
    ])
  elif test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(model.input_size, model.scale_size)
    ])
    
  #Required transformations
  transform = torchvision.transforms.Compose([Transforms.Resize(model.scale_size, interpolation = Image.BILINEAR),
                      Transforms.CenterCrop(model.input_size),
                                
                      ToTorchFormatTensor(div= arch != 'BNInception'),
                      GroupNormalize(model.input_mean, model.input_std),])
  
  # Start looping on frames received from webcam
  softmax = torch.nn.Softmax()
  nn_output = torch.tensor(np.zeros((1, 101)), dtype=torch.float32).cuda()
  frame_count = 0
  try:
    frameobj = Frames_rcv(ip,port)
    frameobj.start()
    while frameobj.is_alive():
        # read each frame and prepare it for feedforward in nn (resize and type)
        frame = frameobj.get_frame()
        frame = Image.fromarray(frame)
        frame = transform(frame).view(1, 3, 224, 224).cuda()
        #print(frame.size())

        # feed the frame to the neural network  
        nn_output += model(frame)

        # vote for class with 25 consecutive frames
        if frame_count % 50 == 0:
            nn_output = softmax(nn_output)
            nn_output = nn_output.data.cpu().numpy()
            preds = nn_output.argsort()[0][-5:][::-1]
            pred_classes = [(idx_to_class[str(pred+1)], nn_output[0, pred]) for pred in preds]

            # reset the process
            nn_output = torch.tensor(np.zeros((1, 101)), dtype=torch.float32).cuda()

        # Display the resulting frame and the classified action
        font = cv2.FONT_HERSHEY_SIMPLEX
        y0, dy = 300, 40
        for i in range(5):
            y = y0 + i * dy
            cv2.putText(orig_frame, '{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]),
                        (5, y), font, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam', orig_frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    frameobj.exit()
    cv2.destroyAllWindows()
  except(KeyboardInterrupt,IOError,Exception) as e:
    frameobj.exit()
    cv2.destroyAllWindows()
    
    

