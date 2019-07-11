from comms_modules.Segmentation import Cap_Process
import argparse
from cv2 import imshow,waitKey,destroyAllWindows
from pathlib import Path
from os.path import abspath

parser = argparse.ArgumentParser(description='Online Action Recognition')
parser.add_argument('--p',dest='port',type = int,default =6666)
parser.add_argument('--ip',dest = 'ip',type=str, default='localhost')
parser.add_argument('--tun',dest = 'tunnel',action='store_true')
parser.add_argument('--rgb',action='store_true')
parser.add_argument('--Ofps',dest= 'old_fps',type= int,default=30)
parser.add_argument('--Nfps',dest= 'new_fps',type= int,default=30)
parser.add_argument('--n1',dest= 'N1',type= int,default=1)
parser.add_argument('--n0',dest= 'N0',type= int,default=0)
parser.add_argument('--k',dest = 'Key_path', type=Path , default=None)
parser.add_argument('--u',dest = 'username' , type = str , default = 'alex039u2')
parser.add_argument('--h',dest= 'hostname' , type = str , default = 'login01.c2.bibalex.org')
parser.add_argument('--passphrase',dest= 'passphrase' , type = str , default = None)
parser.add_argument('--q',dest='encode_quality',type=int,default = 90)
parser.add_argument('--r',dest='reset_threshold',type=int,default = 50)
parser.add_argument('--v',dest='vflag',action='store_true')
parser.add_argument('--vframes',type=int,default = 4)
args = parser.parse_args()


def send_test():
	if bool(args.Key_path):
		if args.Key_path.is_file():
			Key_path = abspath(args.Key_path)
		else:
			print("Error: The Key_path is wrong")
			return 
	else:
		Key_path = args.Key_path

	try :
		port=input("Port : ")
		port =int(port)
	except ValueError:
		port = args.port

	try:
		id = 0
		capture =Cap_Process(fps_old=args.old_fps,fps_new=args.new_fps,id_=id
			,port=port,ip=args.ip,Tunnel=args.tunnel,rgb=False,N1=args.N1,N0=args.N0
			,hostname=args.hostname,username=args.username,Key_path=Key_path,passphrase=args.passphrase
			,reset_threshold=args.reset_threshold,encode_quality=args.encode_quality,vframes=args.vframes,vflag=args.vflag)

		capture.join()
	except (KeyboardInterrupt,IOError,OSError)as e:
		pass
	finally:

		capture.close()
		destroyAllWindows()
if __name__ == '__main__':
	send_test()
