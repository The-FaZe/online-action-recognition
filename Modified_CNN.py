import torchvision
from torch.nn.init import normal_, constant_
from torch import nn
import numpy as np
import torch
from basic_ops import ConsensusModule, Identity
from transforms import *

#nn.Moudle is a base class, the model class should subclass this one
class TSN_model(nn.Module):                                             
  
    def __init__ (self, num_classes, num_segments, modality,
                  consensus_type='avg', base_model_name='resnet18',
                  new_length=None, before_softmax=True, dropout=0.8,
                  crop_num=1, partial_bn=True):
        
        #Excute all nn.Moudle __init__ fuction stuff before anything as a base class.
        super(TSN_model, self).__init__()                                          

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.modality = modality
        self.base_model_name = base_model_name
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax                                        
        self.dropout = dropout
        self.crop_num = crop_num                                                                                                       
        self.partial_bn = partial_bn

        if not before_softmax and consensus_type != 'avg':                          
            raise ValueError("Only avg consensus can be used after Softmax")
        
        #Setting the number of frames picked from each segments 
        if new_length is None:                                               
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
            
        print(("""
          Initializing TSN with base model: {}.
          TSN Configurations:
              input_modality:     {}
              num_segments:       {}
              new_length:         {}
              consensus_module:   {}
              dropout_ratio:      {}
               """.format(base_model_name, self.modality, self.num_segments,
               self.new_length, self.consensus_type, self.dropout)))
        
        #Prepare the standard model
        print('Load and modify the standard model FC output layer')
        self.prepare_model(base_model_name, num_classes)
        print('Done. Loading and Modifying \n ---------------------------------------------------')
        
        
        if self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGBDiff model")
            self.base_model = self.Modify_RGBDiff_Model(self.base_model, keep_rgb=False)
            print("Done. RGBDiff model is ready.")
        
        #Creating Consensus layer (Only 'avg' and 'identity' are available)
        self.consensus = ConsensusModule(consensus_type)                    
        
        #Creating softmax Layer if necessary
        if not self.before_softmax:                                         
            self.softmax = nn.Softmax()
            
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)            
    
    def prepare_model(self, base_model_name, num_classes):
        """
        base_model: string contains the model name 
        this function is used to modify the last layer (fully connected layer)
        for a given architecture to suit our dataset number of actions.
        """
        #add other architectures later
        if 'resnet' in base_model_name:
            #Load pretrained model
            self.base_model = getattr(torchvision.models, base_model_name)(pretrained=True)     
            self.last_layer_name = 'fc'
            #set the input size for the model
            self.input_size = 224                                                               
            self.input_mean = [0.485, 0.456, 0.406]                                            
            self.input_std = [0.229, 0.224, 0.225]                                               

            #There's no point of substarct means from RGBDiff frames
            if self.modality == 'RGBDiff':
                #[0.485, 0.456, 0.406 , 0, 0, 0, 0, 0,.....]
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length  
                #Expand the list with the average 0.452                     
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length 
                
        #BNInception doesn't exist in torchvision models, so we have to get it from net folder
        elif base_model_name == 'BNInception':
            import net
            self.base_model = net.bn_inception()
            self.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            
            #BNInception takes input in range of (0~255), so only mean subtraction should be used in preprocessing.
            #This is different from ResNet models, which take input in the range of (0~1).
            if self.modality == 'RGBDiff':
              self.input_mean = self.input_mean * (1+self.new_length)
              
        else:
            raise ValueError('Unknown base model: {}'.format(base_model_name))
        
        #Get the input size for the last layer of CNN
        features_dim = getattr(self.base_model, self.last_layer_name).in_features                       
        
        #In case of no dropout,Only nn.Linear will be added
        if self.dropout == 0:
            setattr(self.base_model, self.last_layer_name, nn.Linear(features_dim, num_classes))
            self.new_fc = None
            print('The modified linear layer is :', getattr(self.base_model, self.last_layer_name))  
            
        #In case of dropout, خnly nn.Dropout will be added and nn.Linear will be prepared to be added later
        else:
            setattr(self.base_model, self.last_layer_name, nn.Dropout(self.dropout))
            self.new_fc = nn.Linear(features_dim, num_classes)
            print('Dropout Layer added and The modified linear layer is :', self.new_fc)
        
        #Modify Wighets of newly created Linear layer
        std=0.001
        if self.new_fc == None:
            normal_(getattr(self.base_model, self.last_layer_name).weight,0,std)
            constant_(getattr(self.base_model, self.last_layer_name).bias,0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias,0)
       
    def Modify_RGBDiff_Model(self, base_model, keep_rgb=False):
      
        """
        This function to update our first conv2d layer with the appropriate
        number of channels to be suited for the number of frames for each segment.
        """
        
        modules = list(self.base_model.modules())

        #check the index for the first conv2d layer
        for i in range(len(modules)):
            if isinstance(modules[i], nn.Conv2d):
                first_conv_idx = i
                break
        
        #A copy from the 1st Conv2d layer
        conv_layer = modules[first_conv_idx]                         
        container = modules[first_conv_idx-1]
        
        #List which its first element is a (64,3,3,3) tensor (Weights),And the second element is Bais tensor
        params = [x.clone() for x in conv_layer.parameters()]    
        #(64,3,3,3) for resnet18
        kernel_size = params[0].size()                               

        #Modify first layer's size and Weights to suit the input in case of RGBDiff
        #(Used to be 5 stacked frames)
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernel_Weights = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernel_Weights = torch.cat((params[0].data,params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),1)
            new_kernel_size = new_kernel_Weights.size()

        #Create new input layer with proper size.
        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels, conv_layer.kernel_size,
                             conv_layer.stride, conv_layer.padding, bias=True if len(params)==2 else False)
        
        #Override the parameters
        new_conv.weight.data = new_kernel_Weights                                   

        if len(params) == 2:
            # add bias if neccessary
            new_conv.bias.data = params[1].data                                     
        
        #remove .weight suffix to get the layer name
        layer_name = list(container.state_dict().keys())[0][:-7]                    
        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        print('The modified 1st layer is',new_conv)
        return base_model
      
      
    def train(self, mode=True):
      """
      this function freezes batch normalization layers except the first one 
      For any more details, please refer to this paper: https://arxiv.org/pdf/1502.03167.pdf
      inputs: mode (True for the training process)
      """
      super(TSN_model, self).train(mode)
      count=0

      #check if partial batch normalization is activated
      if self.enable_pbn:
          for m in self.base_model.modules():
              if isinstance(m, nn.BatchNorm2d):
                  #freeze the layers except the first one
                  if not self.partial_bn or count > 0:
                      m.eval()  
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False
                  else:
                      count=+1
                      continue
                      
                      
    def extract_rgbDiff(self,RGB_tensor,keep_rgb=False):
      """
      This function for subtracting consecutive frames to obtain RGB difference
      frames. the length of frames for each segment is usually set to be 5
      (you can change it but this gave the best accuracy in the paper).
      RGB_tensor : Tensor contian all frames --Size(Batch_size,num_segments*new_length*3,H,W)
      keep_rgb   : Boolean True(Keep an RGB frame [RGB, RGBDiff, RGBDiff, RGBDiff....])
                           False(All frames are RGBDiff)
      RGBDiff_tensor :Tensor in shape of (Batch_size,num_segments,new_length,3,H,W)
      """
      #Reshape the tensor to 
      #(batch_size, Num of segments, Number of picked frames, C, H, W)
      RGB_tensor = RGB_tensor.view((-1 , self.num_segments , self.new_length+1 , 3 ) + RGB_tensor.size()[2:])
      
      #keep_rgb is a trial from the author to improve accuracy.
      #it is kept false as it only leads to marginal increase.
      if keep_rgb:
          RGBDiff_tensor= RGB_tensor.clone()
      else:
          RGBDiff_tensor = RGB_tensor[:, :, 1:, :, :, :].clone()

      #Generate RGBDiff frames
      #if keep_rgb is set to True, then we will use two streams,
      #one for RGB and one for RGB Diff, so we have to leave the first frame
      #in RGB_tensor non-subtracted
      for x in reversed(list(range(1, self.new_length + 1))):
          if keep_rgb:
              RGBDiff_tensor[:, :, x, :, :, :] = RGB_tensor[:, :, x, :, :, :] - RGB_tensor[:, :, x - 1, :, :, :]
          else:
              RGBDiff_tensor[:, :, x - 1, :, :, :] = RGB_tensor[:, :, x, :, :, :] - RGB_tensor[:, :, x - 1, :, :, :]

      return RGBDiff_tensor
     
    def partialBN(self, enable):
        self.enable_pbn = enable  
                
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
                
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
                    
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult':1, 'decay_mult': 1,'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,'name': "BN scale/shift"},
        ]
      
      
    def forward(self, input): 
        #input size (Batch_size,num_segments*new_length*3,W,H)
        #Total num of channels (3 in RGB and 15 in RGBDiff)                                                  
        sample_len = 3* self.new_length                                         

        if self.modality == 'RGBDiff':
            #Get RGBDiff Tensor in shape of (Batch_size,Num of segments,Number of frames(eg:5),3,H,W)                                          
            input = self.extract_rgbDiff(input)
        
        #Reshape the input to be in shape of (Batchsize*num_segments,new_lenghth*3,H,W) to suit the model.
        input = input.view((-1, sample_len) + input.size()[-2:])
        #print('input after reshape: ',input.size())
        FProp = self.base_model(input) 
        #If the dropout layer is added to the model then there's one more layer to propagate through
        if self.dropout > 0:
            FProp = self.new_fc(FProp)
        #Propagate through softmax too
        if not self.before_softmax:
            FProp = self.softmax(FProp)
        #the output of CNN will be (Num of segments*Batchsize, Num. of classes)
        #So, we have to reshape it to (Batchsize,Num_Segment,Num_classes)
        FProp = FProp.view((-1, self.num_segments) + FProp.size()[1:])
        #Consensus on dim=1 [Consensus the output on segment dim]
        output = self.consensus(FProp)
        return output.squeeze(1)      
        
    @property
    def crop_size(self):                                                                               
        return self.input_size

    @property
    def scale_size(self):                                                                              
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        if self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
