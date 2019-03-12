import Streaming
import Segmentation
import Network
import threading
import multiprocessing as mp
import cv2
from time import sleep
def send_test():
	try:
		id = 0
		capture =Segmentation.Cap_Process(fps_old=1,fps_new=1,id_=id
			,port=6666,ip="192.168.1.2",Tunnel=True,rgb=False)
		while capture.is_alive():
			frame = capture.get()
			if frame is True :
				break
			cv2.imshow('frame',frame)
			cv2.waitKey(10)
	except (KeyboardInterrupt,IOError,OSError)as e:
		pass
	finally:

		capture.close()
		cv2.destroyAllWindows()
if __name__ == '__main__':
	send_test()
