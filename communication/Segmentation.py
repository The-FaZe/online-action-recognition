# a class to determine the decision array
from random import shuffle
import numpy as np
from math import gcd
import threading
import cv2
import multiprocessing as mp
from collections import deque
from communication import Network
from communication import Streaming
from TopN import Top_N
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
        self.cond_ = threading.Condition()
        self.queue_ = deque()
        self.exit_ = False

    def get(self):
        with self.cond_:
            while not len(self.queue_)  :
                self.cond_.wait()
                if self.exit_:
                    return 0
            item = self.queue_.popleft()
        return item
        
    def put(self,item):
        with self.cond_:
            self.queue_.append(item)
            self.cond_.notifyAll()

    def close(self):
        self.exit_=True
        with self.cond_:
            self.cond_.notifyAll()
        self.queue_.clear()
        
        

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


class Cap_Process(mp.Process):
    
    def __init__(self,fps_old,fps_new,id_,port,ip="",Tunnel=True,rgb=True):
        self.frames = mp.Queue(0)
        self.key = mp.Value('b',True)
        self.rgb = rgb
        self.index =decision(fps_old,fps_new)
        self.id_ =id_
        self.port = port
        self.ip = ip
        self.Tunnel = Tunnel
        mp.Process.__init__(self)
        self.start()
    def run(self):
        try:
            client = Network.set_client(port=self.port,
                ip=self.ip,Tunnel=self.Tunnel)
            client2 = Network.set_client(port=self.port,
                ip=self.ip,Tunnel=self.Tunnel)
            vid_cap = cv2.VideoCapture(self.id_)
            success, frame_ = vid_cap.read()
            if not success:
                vid_cap.release()
                send_frames.close()
                self.frames.put(True)
                return
            print("The camera is detected")

            classInd_file = 'UCF_lists/classInd.txt' #text file name
            top5_actions = Top_N(classInd_file)

            send_frames = Streaming.send_frames_thread(client)

            rcv_results = Streaming.rcv_results_thread(connection=client2)
            score = ();
            init = 0
            while (success and self.key.value and send_frames.isAlive() and rcv_results.isAlive()):
                if self.index.index():
                    rcv_results.add()
                    if self.rgb:
                        frame_= frame_[...,::-1]    # Converting from BGR to RGB  
                    send_frames.put(cv2.resize(frame_,(224,224)))
                    count,status,score_=rcv_results.get()
                    if len(score_[0]):
                        init = 1
                        top5_actions.import_indecies_top_N_scores(score_)
                    if len(status):
                        s1 ="Delay of the processing is "+str(count)+" fps"
                        s2 = "The number of waiting frames in buffer is "+str(self.frames.qsize())+" frame"
                        s3 = "the rate of sending frames is "+str(status[0])+" fps"
                        s4 = "The rate of sending data is "+str(status[1])+" KB/s"
                        s = (s1,s2,s3,s4)
                        add_status(frame_,s=s)
                    if init:
                        top5_actions.add_scores(frame_)
                    self.frames.put(frame_)
        
                success, frame_ = vid_cap.read()
        except (KeyboardInterrupt,IOError,OSError) as e :
            pass


        finally:
            self.frames.put(True)
            send_frames.close()
            rcv_results.close()
            client.close()
            print("The program cut the connection")
            vid_cap.release()
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
        self.queue = np.array([])
        self.max = max
    def mean(self,inp,dim=2):
        if self.queue.size is 0:
            self.queue = np.array(inp)
            self.queue = np.expand_dims(self.queue , axis=0)
        else:
            self.queue = np.concatenate((np.expand_dims(inp , axis=0)
                                         ,self.queue), axis=0)
            if (self.queue.shape[0] >= self.max):
                self.queue = self.queue[0:self.max-1]
        return np.mean(self.queue, axis=0)




# A method to add status on the image the frame_ is the incoming image and s1,s2,s3 are the txt to put on the image

def add_status(frame_,s=(),x=5,y=10,i=20,font = cv2.FONT_HERSHEY_SIMPLEX
    ,fontScale = 0.4,fontcolor=(255,255,255),lineType=1):
    c = 0
    for s_ in s:
        l = i*c
        cv2.putText(frame_,s_, (x,y+l) 
            ,font, fontScale, fontcolor, lineType)
        c += 1
