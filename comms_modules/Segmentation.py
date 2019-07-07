# a class to determine the decision array
from random import shuffle
import numpy as np
from math import gcd
import threading
import cv2
import multiprocessing as mp
from collections import deque
from itertools import cycle
from . import Network,Streaming
from .TopN import Top_N
# A class to generate random index that segment the real time stream
# then pick snippets out of every segment in real time behaviour 
class decision():
    
    #A method to construct the decision array as soon as it's called
    
    def __init__(self,fpso,fpsn):
        GCD  = gcd(fpso,fpsn)           #Calculating greatest common divisor
        L    = fpso//GCD                #The number of frames in 1 segment
        n    = fpsn//GCD                #The number of  snippets in 1 segment

        #Decision array is to determine keeping the frame or dropping it,
        #that number of ones determines number of (selected frames(snippets) - the very first frame) in a seg
        # converting the numpy array to list datatype (more efficient in this case)
        
        self.array    = np.append(np.zeros(L-n),np.ones(n-1)).tolist()
        shuffle(self.array)             #Shuffling the instants those frames are captured in
        self.array.append(1)            #Adding the first frame index in the begining of the of the Decision array
        self.backup = self.array.copy() #the back-up for the list for generating a new one when it's totally consumed 

    # A method to get the current index and generating new array when it's totally consumed 
    def index(self):
        index_ = self.array.pop()           # getting the index from the end of the array 
        if not len(self.array):             # If there is no elements in the list start generating new one 
            self.array = self.backup.copy() # Copy the elements from the backup list
            shuffle(self.array)             # Shuffle the array to select in diferent locations 
        return index_




class thrQueue():
    
    def __init__(self):
        self.cond = threading.Condition(threading.RLock())
        self.count_ = 0
        self.queue = deque()
        self.clear_ = False
        self.reset_ = False

    def get(self):
        with self.cond:
            if self.clear_:
                item = 0
            elif self.reset_:
                item = None
            else:
                while(self.count_ <= 0):
                    self.cond.wait()
                    if (self.clear_|self.reset_):
                        break
                if(self.clear_|self.reset_):
                    item = self.get()
                else:
                    item = self.queue.popleft()
                    self.count_ -= 1
        return item
        
    def put(self,item):
        with self.cond:
            if (self.clear_|self.reset_):
                pass
            else:
                self.queue.append(item)
                self.count_ += 1
                self.cond.notify() 


    def close(self):
        with self.cond:
            self.clear_ = True
            self.queue.clear()
            self.cond.notify()

    def reset(self):
        with self.cond:
            self.reset_=True
            self.queue.clear()
            self.cond.notify()
    
    def confirm(self):
        with self.cond:
            self.reset_ = False
            self.count_ = 0


    def qsize(self):
        with self.cond:
            qs = self.count_
        return qs

        
        
"""
class Cap_Thread(threading.Thread):
    
    def __init__(self,fps_old,fps_new,port,id_=0):
        self.frames = thrQueue()
        self.key = True
        threading.Thread.__init__(self)
        self.index =decision(fps_old,fps_new)
        self.id_ =id_
        self.start()
        
    def run(self):

        vid_cap = cv2.VideoCapture(self.id_)
        success, frame_ = vid_cap.read()
        
        if not success:
            print ('No Camera is detected ')
            vid_cap.release()
            self.frames.put(True)
            return
        print("The camera is detected")
        while (success and self.key):
            
            if self.index.index():
                self.frames.put(frame_)
                
            success, frame_ = vid_cap.read()
        vid_cap.release()
        print('The secound process is terminated ')

    def get(self,rgb = True):
        frame_ = self.frames.get()      # Getting a frame from the queue
        if rgb:
            frame_ = frame_[...,::-1]    # Converting from BGR to RGB 
        return frame_ 			# Returning the frame as an output

    def close(self):
        self.key = 0   # breaking the capture loop
        self.join()     # waiting for the capture thread to terminate smoothly
        self.frames.close()
        print ('No More frames to capture ')

"""
class Cap_Process(mp.Process):
    
    def __init__(self,fps_old,fps_new,id_,port,ip="0.0.0.0",reset_threshold=30,encode_quality=90,Tunnel=True,rgb=True,N1=1,N0=0
        ,hostname=None,username=None,Key_path=None,passphrase=None):
        self.frames = mp.Queue(0)
        self.key = mp.Value('b',True)
        self.rgb = rgb
        self.index =decision(fps_old,fps_new)
        self.reset_threshold=reset_threshold
        self.encode_quality = encode_quality
        self.id_ =id_
        self.port = port
        self.ip = ip
        self.Tunnel = Tunnel
        self.N1=N1
        self.N0=N0
        self.hostname=hostname
        self.username=username
        self.Key_path=Key_path
        self.passphrase=passphrase
        mp.Process.__init__(self)
        self.start()
    def run(self):
        active_reset = False
        client,transport = Network.set_client(ip=self.ip,port=self.port,numb_conn=2,Tunnel=self.Tunnel,
            hostname=self.hostname,username=self.username,Key_path=self.Key_path,passphrase=self.passphrase)
        if client is None:
            self.frames.put(True)
            return 0
        
        try:

            cam_cap = cv2.VideoCapture(self.id_)
            success, frame_ = cam_cap.read()
            itern = cycle(self.N1*(1,)+self.N0*(0,))
            if not success:
                cam_cap.release()
                send_frames.close()
                self.frames.put(True)
                return
            print("The camera is detected")

            classInd_file = 'UCF_lists/classInd.txt' #text file name
            top5_actions = Top_N(classInd_file)

            send_frames = Streaming.send_frames_thread(connection=client[0],reset_threshold=self.reset_threshold,encode_quality=self.encode_quality)

            rcv_results = Streaming.rcv_results_thread(connection=client[1])
            score = ();
            initialized = False
            s3 = "No Status Received"
            s4 = "No Status Received"
            while (success and self.key.value and send_frames.isAlive() and rcv_results.isAlive()):
                if not send_frames.Actreset() :
                    if self.index.index():
                        if next(itern):
                            rcv_results.add()
                            if self.rgb:
                                frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)    # Converting from BGR to RGB  
                            send_frames.put(cv2.resize(frame_,(224,224)))


                    count,status,score_,NoActf,test,New_out=rcv_results.get()
                    _=send_frames.status()
                    if New_out[1]:
                        initialized = True
                        top5_actions.import_indecies_top_N_scores(score_)
                    if New_out[0]:
                        s3 = "the rate of sending frames is {0:.2f} fps".format(status[0])
                        s4 = "The rate of sending data is {0:.2f} KB/s".format(status[1])
                    s1 = "UP/Down delay {0:1d} frame".format(int(count))
                    #s2 = "The number of waiting frames in buffer is {0:.2f} frame ".format(self.frames.qsize())
                    s5 = "Buffered frames {0:1d} frame".format(int(send_frames.frames.qsize()))

                    s = (s1,s3,s4,s5)
                    frame_=remove_bounderies1(frame_)
                    frame_=cv2.pyrUp(frame_)
                    print(frame_.shape)
                    alpha = 0.3
                    font=cv2.FONT_HERSHEY_TRIPLEX
                    y=frame_.shape[0]-15
                    add_status(frame_,s=s,font=font,thickness=5,lineType=2,fontScale=1.25,fontcolor=(0,0,0),box_flag=True,boxcolor=(200,200,200,0.1),x=15,y=30,alpha=alpha)

                    if test and NoActf:
                        top5_actions.add_scores(frame_,font=font,thickness=5,lineType=2,fontScale=1.6,fontcolor=(0,0,255),x=5,y=y,box_flag=True,boxcolor=(200,200,200),final_action_f=True,alpha=alpha,x_mode='center')
                    elif NoActf:
                        add_status(frame_,s=("No Assigned Action Detected",),x=5,y=y,font=font,thickness=5,fontScale=1.6,fontcolor=(0,0,255),box_flag=True,boxcolor=(200,200,200),lineType=2,alpha=alpha,x_mode='center')
                    elif initialized:
                        top5_actions.add_scores(frame_,font=font,thickness=5,lineType=2,fontScale=1.6,fontcolor=(0,0,0),x=5,y=y,box_flag=True,boxcolor=(200,200,200),final_action_f=True,alpha=alpha,x_mode='center')
                    else :
                        add_status(frame_,s=('Start Recognition',),x=5,y=y,font=font,thickness=1,lineType=2,fontScale=1.6,fontcolor=(0,0,0),boxcolor=(200,200,200),alpha=alpha,x_mode='center')
                else:
                    initialized = False
                    rcv_results.reset()
                    add_status(frame_,s=('Reseting',),x=5,y=y,font=font,thickness=1,lineType=2,fontScale=1.6,fontcolor=(0,0,255),boxcolor=(200,200,200),alpha=alpha,x_mode='center')
                
                self.frames.put(frame_)
                success, frame_ = cam_cap.read()
        except (KeyboardInterrupt,IOError,OSError) as e :
            pass


        finally:
            self.frames.put(True)
            send_frames.close()
            rcv_results.close()
            print("The program cut the connection")
            cam_cap.release()
            if bool(transport):
                transport.close()
            print("The program broke the connection to the camera")



    def get(self):
        frame_ = self.frames.get() 
		# Returning the frame as an output
        return frame_

    def close(self):
        self.key.value = False   # breaking the capture loop
        self.join()     # waiting for the capture thread to terminate smoothly
        self.frames.close()
        print ('No More frames to capture ')
            

# A module to generate the mean of the input in real time with window == max

class mean():
    def __init__(self,max = 30):
        self.queue = deque(max*[False],maxlen=max)
        self.max = max
        self.init = False
        self.out = False
        self.c = 0
    def mean(self,inp):
        self.c = min(self.c+1,self.max)
        carry = self.queue.popleft()
        inp = np.array(inp)
        self.queue.append(inp)
        self.out += inp-carry
        return self.out/self.c
    def mean_temp(self,inp):
        c = min(self.c+1,self.max+1)
        inp = np.array(inp)
        return (self.out+inp)/c

# A method to add status on the image the frame_ is the incoming image and s1,s2,s3 are the txt to put on the image

def add_status(frame_,s=(),x=5,y=12,font = cv2.FONT_HERSHEY_SIMPLEX
    ,fontScale = 0.4,fontcolor=(255,255,255),lineType=1,thickness=3,box_flag=True
    ,alpha = 0.4,boxcolor=(129, 129, 129),x_mode=None):
    l = 0
    if x_mode is 'center':
        x=frame_.shape[1]//2
    elif x_mode is 'left':
        x=frame_.shape[1]
    for s_ in s:
        origin=np.array([x,y+l])
        y_c = add_box(frame=frame_ ,text=s_ ,origin=origin, font=font, fontScale=fontScale
            ,thickness=thickness ,alpha=alpha ,enable=box_flag,color=boxcolor,x_mode=x_mode)
        cv2.putText(frame_,s_, tuple(origin) 
            ,font, fontScale, fontcolor, lineType,thickness)
        l += y_c


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


def remove_bounderies(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    print(x,y,w,h)
    return frame[y:y+h,x:x+w]

def remove_bounderies1(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    thresh_binary_y_wise = (thresh.sum(1) > 0)
    r_0=(np.logical_xor(thresh_binary_y_wise[1:],thresh_binary_y_wise[0:-1])).nonzero()[0]
    r_0=sorted(r_0,reverse=True)
    if len(r_0) == 0 :
        return frame
    location=list(map(lambda x : (not(thresh_binary_y_wise[0:x].sum()),not(thresh_binary_y_wise[x+1:].sum())) , r_0))# (0,1) top border , (1,0) botom border (1,1) pass
    index_ = np.array(list(map(lambda x,y :(x[0] | x[1])*y , location,r_0))).nonzero()[0]

    for index in index_:
        #print(location[index])
        if location[index] == (False,True):
            frame = frame[0:r_0[index]]
        elif location[index] == (True,False):
            frame = frame[r_0[index]:]
        else:
            pass
    return frame