import Streaming
import Network
import threading
import multiprocessing as mp
from TopN import Top_N
import numpy as np
def test_server():
	try:
		Tunnel_ = False
		classInd_file = 'UCF_lists/classInd.txt'
		top5_actions = Top_N(classInd_file)
		conn,T_thr = Network.set_server(port=6666,Tunnel=Tunnel_,n=1)
		rcv_frames = Streaming.rcv_frames_thread(connection=conn[0])
		send_results = Streaming.send_results_thread(connection=conn[1])
		c = 0;
		while (rcv_frames.isAlive() and send_results.isAlive()):
			frame,status = rcv_frames.get()
			if frame is 0:
				break          
			if c % 50 == 0 and c != 0:
				scores = np.random.random(101)
				top5_actions.import_scores(scores)
				index,_,scores = top5_actions.get_top_N_actions()
				send_results.put(status=status,scores=(*index,*scores))
			else:
				send_results.put(status=status)
			c +=1
	except (KeyboardInterrupt,IOError,OSError) as e:
		pass
	finally:
		rcv_frames.close()
		send_results.close()
		conn[0].close()
		conn[1].close()
		if Tunnel_:
			T_thr.terminate()
if __name__ == '__main__':
	test_server()
