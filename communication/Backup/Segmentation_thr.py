import cv2
import numpy as np
import random
import threading
from math import gcd

#the presented code is a mutithread class consists of 2 modules 
#	The first module is run on a parallel thread to the main thread and tasked to select the snippets in a random way 
#	then saving the selected snippets into the memory.
#The presented code is capped at the actual FPS of the camera.
class FrameCap(threading.Thread): # defining a thread class
    frames = np.asarray([])       # Allocating a name for the captured frames
    key   = True                  # Key to kill the process in the 2nd thread(parallel thread to the main) using the main thread.
    event = threading.Event()     # An Event for communication between threads
    initial_ =threading.Event()   # For decalring intialization 
    success = True                # A flag to insures the success of capture
    flag = False                  #A flag to check initialization
    def __init__(self, fps_old, fps_new, id_):
        threading.Thread.__init__(self)
        self.fps_old = fps_old       #The FPS of the camera
        self.fps_new = fps_new       #The FPS of the output
        self.id = id_                #The ID of the camera


    # A module to select and save the selected frames into an array (runs in a parallel thread to the main thread of the main code)
    def run(self):
	# Defining names
        fpso = self.fps_old   	    #The FPS of the camera
        fpsn = self.fps_new 	    #The FPS of the output
        GCD  = gcd(fpso,fpsn)	    #Calculating greatest common divisor
        L    = fpso//GCD            #The number of frames in 1 segment
        n    = fpsn//GCD	    #The number of 	snippets in 1 segment
	#Decision array is to determine keeping the frame or dropping it,
        #that number of ones determines number of (selected frames(snippets) - the very first frame) in a seg
        i    = np.append(np.zeros(L-n),np.ones(n-1))
        np.random.shuffle(i)				 #Shuffling the instants those frames are captured in
        i    = np.append(1,i)				 #Adding the first frame index in the begining of the of the Decision array
        k    = i.copy() 				 #Copying those frames into another name
        vid_cap = cv2.VideoCapture(self.id)		 #Creating the object vid_cap for the camera capturing
        _       = vid_cap.open(self.id)			 #Opening the camera
        self.success, self.frames = vid_cap.read()	 #capturing and selecting the first frame to record
        self.initial_.set()                              # Waking up the the main thread by initialization 
        self.flag = True                                 # The flag to insures it woke up from the flag not from the time out
        if not self.success:                             # going out of the thread if there is no camera
            print ('No Camera is detected  ')
            vid_cap.release()
            return
	#adding one more dimension to the main frames matrix to concatinate the next frames onto
        self.frames = np.expand_dims(self.frames, axis=0)
	#Capturing loop designed to break, If the key is set to 0 or there's an error accessing the camera
        while (self.key):		                 
            i = i[1:]					 #Droping the first index of the Decision array
            #If the decision array is an empty array it will copy the saved deicision array and shuffle its elements
            if not i.size:
                i = k.copy()
                np.random.shuffle(i)
            self.success, frame_ = vid_cap.read()	 # Capturing and decoding the next frame
            if not self.success:                         # Closing the camera after breaking the loop if there is no frames to capture
                vid_cap.release()			 
                return
            if i[0]:		# Taking a decision to drop or concatente it onto the frames name of the class "FrameCap"
                self.frames = np.concatenate((self.frames, np.expand_dims(frame_, axis=0)), axis=0)
                self.event.set()
        vid_cap.release()				 # Closing the camera after breaking the loop
        return
    #This module is to fetch first frame which saved in the memory then erasing it(run in the main thread with the main code)
    def get_frame(self,rgb = True):
        self.event.clear()
        while ((not(self.frames.size)) and self.success):
          self.event.wait()
        frame_ = self.frames[0]				#Fectching the frame from the top of the array
        self.frames = self.frames[1:]		    	#Delecting the fetched frame
        if rgb:
            b,g,r  = cv2.split(frame_)                  # get b,g,r
            frame_ = cv2.merge([r,g,b])       	        # switch it to rgb
        return frame_ 					#Returning the frame as an output
    
    def initial(self):                          #initial module to insures that the class has been started taking data
        while not self.flag:                    # insures the waking up of the thread isn't from the internal timeout
            self.initial_.wait()

#For testing
def test():
    frame = FrameCap(8,6,0)                     # setting up the object
    frame.start()                               # initializing the capture thread
    try:
        frame.initial()                         # Waiting for intialization 
        while frame.success:                    # Real time processing loop 
            frame_ = frame.get_frame(False)     # Getting a fraf in form of BGR
            frame_ = cv2.flip(frame_,0)         # The rest of code here(Any kind of processing is here)
            cv2.imshow('frame',frame_)          # The rest of code here(Any kind of processing is here)
            cv2.waitKey(30)
        print ('No More frames to capture  ')   # Printing there is no frames when breaking out of the loop
        frame.key = 0                           # breaking the capture thread
        frame.join()                            # waiting for the capture thread to terminate
        cv2.destroyAllWindows()                 # clearing the windows
    except KeyboardInterrupt:
        frame.key = 0                           # breaking the capture thread
        frame.join()                            # waiting for the capture thread to terminate
        cv2.destroyAllWindows()                 # clearing the windows
        print('The program has been terminated')
#Main For Testing
if __name__ == '__main__':
    test()
