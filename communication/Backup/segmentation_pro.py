import cv2
import numpy as np
import random
import multiprocessing as mp
from math import gcd
from time import sleep
#the presented code is a mutithread class consists of 2 modules 
#	The first module is run on a parallel thread to the main thread and tasked to select the snippets in a random way 
#	then saving the selected snippets into the memory.
#The presented code is capped at the actual FPS of the camera.
class FrameCap(mp.Process):       # defining a thread class

    def __init__(self, fps_old, fps_new, id_):
        self.frames = mp.Queue(0)          # Allocating a name for the captured frames
        self.key  = mp.Value('b',True)     # Key to kill the process in the 2nd thread(parallel thread to the main) using the main thread.
        mp.Process.__init__(self,daemon=False)
        self.fps_old = fps_old             #The FPS of the camera
        self.fps_new = fps_new             #The FPS of the output
        self.id = id_                      #The ID of the camera
        
    # a method to determine the decision array
    def decision(self):
        fpso = self.fps_old   	    #The FPS of the camera
        fpsn = self.fps_new 	    #The FPS of the output
        GCD  = gcd(fpso,fpsn)	    #Calculating greatest common divisor
        L    = fpso//GCD            #The number of frames in 1 segment
        n    = fpsn//GCD	    #The number of  snippets in 1 segment
	#Decision array is to determine keeping the frame or dropping it,
        #that number of ones determines number of (selected frames(snippets) - the very first frame) in a seg
        i    = np.append(np.zeros(L-n),np.ones(n-1))
        np.random.shuffle(i)				 #Shuffling the instants those frames are captured in
        i    = np.append(1,i)				 #Adding the first frame index in the begining of the of the Decision array
        return i

    # A module to select and save the selected frames into an array (runs in a parallel thread to the main thread of the main code)
    def run(self):
        i = self.decision()                              #Getting Index array (decision array)
        k = i.copy() 				         #Back up this array
        vid_cap = cv2.VideoCapture(self.id)		 #Creating the object vid_cap for the camera capturing
        success, frame_ = vid_cap.read()	         #capturing and selecting the first frame to record
        if not success:                                  # going out of the thread if there is no camera
            print ('No Camera is detected ')
            vid_cap.release()
            self.frames.put(True)
            return
        #Capturing loop designed to break, If the key is set to 0 or there's an error accessing the camera
        while (self.key.value and success):
            if i[0]:		        # Taking a decision to drop or concatente it onto the frames name of the class "FrameCap"
                self.frames.put(frame_)
            i = i[1:]					 #Droping the first index of the Decision array(consuming the array)
            #If the decision array is an empty array it will copy the saved deicision array and shuffle its elements
            if not i.size:
                i = k.copy()
                np.random.shuffle(i)
            success, frame_ = vid_cap.read()	    # Capturing and decoding the next frame
        vid_cap.release()				 # Closing the camera after breaking the loop
        print('The secound process is terminated ')
        self.frames.put(True)
        return
    
    #This module is to fetch first frame which saved in the memory then erasing it(run in the main thread with the main code)
    def get_frame(self,rgb = True):
        frame_ = self.frames.get(True,30)   
        if rgb:
            b,g,r  = cv2.split(frame_)                  # get b,g,r
            frame_ = cv2.merge([r,g,b])       	        # switch it to rgb
        return frame_ 					#Returning the frame as an output

    
#For testing
def main(fun,args=()):
    frame = FrameCap(8,6,0)                     # setting up the object
    frame.start()                               # initializing the capture thread
    try:
        while frame.is_alive():                    # Real time processing loop
            frame_ = frame.get_frame(False)     # Getting a fraf in form of BGR
            if frame_ is True:
                break
            
            fun(frame_)
            
        print ('No More frames to capture ')   # Printing there is no frames when breaking out of the loop
        frame.frames.close()
        frame.join()                            # waiting for the capture thread to terminate
        print('The programe is exiting ')
        cv2.destroyAllWindows()                 # clearing the windows
        sleep(2)
    except KeyboardInterrupt:
        frame.key.value = False                 # breaking the capture thread
        frame.join()                            # waiting for the capture thread to terminate
        frame.frames.close()
        cv2.destroyAllWindows()                 # clearing the windows
        print('The program has been terminated ')
def test(frame_,):
    frame_ = cv2.flip(frame_,0)         # The rest of code here(Any kind of processing is here)
    cv2.imshow('frame',frame_)          # The rest of code here(Any kind of processing is here)
    cv2.waitKey(30)
#Main For Testing
if __name__ == '__main__':
    main(test)
