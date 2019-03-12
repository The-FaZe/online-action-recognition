import numpy as np
import multiprocessing as mp
from time import time
from TCP import  Frames_rcv
import cv2
#For testing
def main():
    try:
        frame = Frames_rcv('192.168.1.112',6666,True)
        frame.start()
        while frame.is_alive():                 # Real time processing loop
            frame_ = frame.get_frame(False)     # Getting a fraf in form of BGR
            #frame_ = cv2.flip(frame_,0)         # The rest of code here(Any kind of processing is here)
            cv2.imshow('frame',frame_)          # The rest of code here(Any kind of processing is here)
            cv2.waitKey(30)
        frame.exit()                               # clearing the windows
        cv2.destroyAllWindows()
    except (KeyboardInterrupt,IOError,Exception,OSError)as e:
        frame.exit()
        cv2.destroyAllWindows()
#Main For Testing
if __name__ == '__main__':
    main()
