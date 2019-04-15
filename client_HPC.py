from comms_modules.Segmentation import Cap_Process
import argparse
from cv2 import imshow,waitKey,destroyAllWindows

parser = argparse.ArgumentParser(description='Online Action Recognition')
parser.add_argument('--p',dest='port',type = int,default =6666)
parser.add_argument('--ip',type=str, default='localhost')
parser.add_argument('--tun',dest = 'tunnel',action='store_true')
parser.add_argument('--rgb',action='store_true')
parser.add_argument('--Ofps',dest= 'old_fps',type= int,default=30)
parser.add_argument('--Nfps',dest= 'new_fps',type= int,default=30)
parser.add_argument('--n1',dest= 'N1',type= int,default=1)
parser.add_argument('--n0',dest= 'N0',type= int,default=0)
args = parser.parse_args()
def send_test():
	try:
		id = 0
		capture =Cap_Process(fps_old=args.old_fps,fps_new=args.new_fps,id_=id
			,port=args.port,ip=args.ip,Tunnel=args.tunnel,rgb=False,N1=args.N1,N0=args.N0)
		while capture.is_alive():
			frame = capture.get()
			if frame is True :
				break
			imshow('frame',frame)
			waitKey(10)
	except (KeyboardInterrupt,IOError,OSError)as e:
		pass
	finally:

		capture.close()
		destroyAllWindows()
if __name__ == '__main__':
	send_test()
