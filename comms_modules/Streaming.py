from . import Network, Segmentation
import multiprocessing as mp
from time import time,sleep
import threading
from cv2 import cvtColor,COLOR_BGR2RGB,IMWRITE_JPEG_QUALITY,imencode
from socket import socket
from struct import pack,unpack,calcsize

class rcv_frames_thread(threading.Thread):
    """
    Class repsonsible for generating a thread responsible for receiving frames 
    the downstream connection socket is passed to it 
    the maximum window for calculating the average speed(mean)
    Status flag
    """

    def __init__(self,connection=socket()
        ,status=True,w_max=30):
        
        self.connection = connection
        
        self.frames = Segmentation.thrQueue()
        
        self.key  = True

        self.status = status

        self.active_reset = False

        self.cond = threading.Condition(threading.Lock())

        threading.Thread.__init__(self)

        if self.status:  # If the flag is true then initialize the mean object 
            self.m = Segmentation.mean(w_max)

        self.start()



    def run(self):
        """
        a method runs on a parallel thread when calling this class
        reponsible for listening and capturing the stream of data on the connection and buffer it into the queue for consuming
        """
        try:
            msglen_sum = 0 #the size of total frames received 
            y = time() #Total time of the receiving
            # The loop to receive frames from the connection
            #self.key is a flag to be used by the main process to break the loop and terminate the parallel process
            while (self.key):

                x = time() # start recording the time of receiving each frame 


                frame_,msglen = Network.recv_frame(self.connection) # receiving a frame

                msglen_sum += msglen # updating the size of the total frames received

                if (msglen is 0):

                    self.active_reset = True
                    
                    self.frames.reset()
                    self.WaitConfirmReset()
                    self.frames.confirm()

                else:
                #adding the (frame received,the size of the frame and the total time it took to receive it) in the shared memory buffer
                    self.frames.put([frame_,msglen,time()-x]) 

        #breaking the connection and terminating the process if there is an error , interruption or connection break 
        except ( KeyboardInterrupt,IOError,OSError)as e:
            pass

        finally:
            print('The secound process is terminated \n',   #calculating and printing the average speed of the connection 
                  'The total average Rate is ',msglen_sum/((time()-y)*1000),'KB/s')

            self.connection.close() #closing the connection if the for loop is broken

            self.frames.close() #declaring that there is no data will be added to the queue from this process
        return

    def ConfirmReset(self):
        """
        a method to confirm that all procedure for reset has been done and it's ready for reactivation 
        """
        with self.cond:

            self.active_reset = False

            self.cond.notify()

        self.connection.sendall(b'\xff')


    def WaitConfirmReset(self):
        """
        a method to block the parallel thread untill the confimation is received for reactivation
        """
        with self.cond:
                while self.active_reset:
                    self.cond.wait()

    def CheckReset(self):
        """
        a method to check the reset conditions contineously
        """
        with self.cond:
            ar =  self.active_reset
        return ar


    #The method is responsible for consuming data from the queue ,decoding it
    # printing status on it and converting it into RGB if desired 
    def get(self,rgb=True):
        """
        The method is responsible for consuming data from the queue ,decoding it
        printing status on it and converting it into RGB if desired using rgb flag
        """
        if self.CheckReset():
            return None,None
        else:
            frame_ = self.frames.get() #blocking until getting the data with time out of 60 s 
            if frame_ is 0:
                return 0,()
            if frame_ is None:
                print(self.CheckReset())
                if self.CheckReset():
                    return None,None
            frame_ , msglen , spf = frame_
            frame_ = Network.decode_frame(frame_) #decoding the frames
            if rgb:
                frame_ = cvtColor(frame_, COLOR_BGR2RGB)    # Converting from BGR to RGB
            [msglen_rate,spf] = self.m.mean([msglen,spf])
            status = (1/spf,msglen_rate/(spf*1000))        
            return frame_ ,status                                # Returning the frame as an output



    # The method is responsible for Quiting the program swiftly with no daemon process
    # the method is to be using in the main process code 
    def close(self):
        '''
        this method responsible for terminating the thread
        '''
        self.key = False # breaking the while loop of it's still on in the parallel process(run)
        self.frames.close() # declaring there is no frames will be put on the shared queue from the main process
        self.join() # waiting for the parallel process to terminate
        print('The program has been terminated ')




class send_frames_thread(threading.Thread):
    """
    a class responsible for generating a thread to send frames across the TCP socket connection
    takes in the upstream socket connection 
    The reset threshold it will reset the connection after accumilating to this threshold
    encoding quality of the frames
    """
    def __init__(self,connection=socket(),reset_threshold=60,encode_quality=90,w_max=30):
        self.key = True
        self.frames = Segmentation.thrQueue()
        self.connection = connection
        self.encode_quality=encode_quality
        self.reset_threshold=reset_threshold
        self.active_reset = False
        self.cond_ = threading.RLock()
        self.latest_spotted_time_measerand = [False,None]
        self.encode_param=[int(IMWRITE_JPEG_QUALITY),encode_quality] # object of the parameters of the encoding (JPEG with 90% quality) 
        if self.status:  # If the flag is true then initialize the mean object 
            self.m = Segmentation.mean(w_max)
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        """
        a method run on a new thread to consume the queued frames adding headers and sending it through TCP connection
        """
        try:
            while (self.key):
                if self.frames.qsize() >= self.reset_threshold:
                    self.active_reset = True
                    self.frames.reset()
                    Network.send_frame(connection=self.connection,img=None,active_reset=self.active_reset)
                    self.frames.confirm()
                else:
                    self.active_reset = False 
                    x = time()
                    with self.cond_:
                        self.latest_spotted_time_measerand=[True,x]
                    msglen=Network.send_frame(connection=self.connection,img=self.frames.get(),active_reset=self.active_reset)
                    with self.cond_:
                        t=time()-x
                        self.m.mean([msglen,t])
                        self.latest_spotted_time_measerand[0]=False


        except(KeyboardInterrupt,IOError,OSError) as e:
            pass

        finally:
            self.connection.close()
            print('sending Frames is stopped')

    def put(self,frame):
        """
        putting data in a queue to be consumed by the new thread and send in order .
        """
        if not self.active_reset:
            _ ,frame = imencode('.jpg',frame,self.encode_param) # encoding the img in JPEG with specified quality 
            self.frames.put(frame)

    def Actreset(self):
        """
        method responsible for checking the reset condition (for ease use)
        """
        return self.active_reset

    def close(self):
        """
        to terminate the thread swiftly
        """
        self.key = False # breaking the while loop of it's still on in the parallel process(run)
        self.frames.close() # declaring there is no frames will be put on the shared queue from the main process
        self.join() # waiting for the parallel process to terminate

    def status(self):
        with self.cond_:
            if self.latest_spotted_time_measerand[0]:
                t=time()-self.latest_spotted_time_measerand[1]
                out=self.m.mean_temp([0,t])
            else :
                out=self.m.mean.out
        return (1/out[1],out[0]/(out[1]*1000)) 


class send_results_thread(threading.Thread):
    """
    this class is responsible for generating a thread to send frames through the socket connection contineously

    Number of wanted top scores 
    number of assignet status to be send 
    and the test flag
    """
    def __init__(self,connection=socket(),nmb_scores=5,nmb_status=2,test=False):

        self.key_ = True
        self.results = Segmentation.thrQueue()
        self.connection = connection
        self.check = (nmb_status,2*nmb_scores,2*nmb_scores+nmb_status,None)
        self.fmb = (">{}f".format(nmb_status)
            ,">{}B{}f".format(nmb_scores,nmb_scores)
            ,">{}f{}B{}f".format(nmb_status,nmb_scores,nmb_scores),None)
        self.test = test
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        """
        the generated thread that sends the data as soon as available
        and adding headers of the application layer
        """
        try:
            while (self.key_):
                result = self.results.get()
                if result is 0:
                    break
                elif result[0]:
                    flag = self.check.index(len(result[0]))
                    flagb = pack(">B", flag | (0x80*result[1]) | (0x40*self.test))
                    self.connection.sendall(flagb)
                    results_ = pack(self.fmb[flag],*result[0])
                    self.connection.sendall(results_)
                    #print(self.results.qsize())
                else:
                    flag = self.check.index(None)
                    flag = flag | (0x80*result[1])
                    flagb = pack(">B", flag)
                    self.connection.sendall(flagb)
                    #print(self.results.qsize())
        except(KeyboardInterrupt,IOError,OSError) as e:
            pass

        finally:
            self.connection.close()
            self.results.close()
            print('sending results is stopped')

    def reset(self):
        '''
        reseting the queue
        '''
        results.reset()
        self.confirm()

    def put(self,status=(),scores=(),Actf=False):
        """
        putting results into the queue
        """
        self.results.put([status + (scores*(Actf|self.test))*bool(status + (scores*(Actf|self.test))) ,bool(not(Actf))])

    def close(self):
        """
        terminating the thread 
        """
        self.key_ = False 
        self.results.close()
        self.join() # waiting for the parallel process to terminate


        

class rcv_results_thread(threading.Thread):
    """
    this calss is responsible for generating a thread to receive results and getting the most recent results out of the downstream socket connection

    The inputs 

    Number of wanted top scores 
    number of assignet status to be send 

    the output 

    the resukts 
    """
    def __init__(self,connection=socket(),nmb_scores=5,nmb_status=2):

        self.key_ = True
        self.connection = connection
        self.fmb = (">{}f".format(nmb_status)
            ,">{}B{}f".format(nmb_scores,nmb_scores)
            ,">{}f{}B{}f".format(nmb_status,nmb_scores,nmb_scores),None)
        self.count = 0
        self.result_ = ()
        self.cond = threading.Condition(threading.Lock())
        self.nmb_scores = nmb_scores
        self.nmb_status = nmb_status
        self.test = False
        self.New_out = [False,False]
        self.NoActf = False
        self.action_index = []
        self.scores = []
        self.status = []

        threading.Thread.__init__(self)
        self.start()


    def run(self):
        """
        The parallel thread that listens on the connection
        and decapsulate the header taking the flags and the information needed.
        """
        try:
            while (self.key_):
                flag = Network.recv_msg(self.connection, 1 , 1)
                flag = int(unpack(">B",flag)[0])
                NoActf = bool(flag & 0x80)
                test = bool(flag&0x40)
                flag = flag & 0x0F
                fmb = self.fmb[flag]
                if bool(fmb):
                    results = Network.recv_msg(self.connection,calcsize(fmb), 2048)
                    results = unpack(fmb,results)
                    self.update(result=results,NoActf=NoActf,test = test)
                else:
                    with self.cond:
                        self.NoActf = NoActf
                        self.test = test
                        self.count -= 1


        except(KeyboardInterrupt,IOError,OSError) as e:
            pass
        finally:
            self.connection.close()
            print('receiving results is stopped')


    def update(self,result,NoActf,test):
        """
        updating to the latest results 
        """
        with self.cond:
            self.NoActf = NoActf
            self.test = test
            if(len(result)==self.nmb_status):
                self.status = result
                self.New_out[0] = True  
            elif(len(result)==2*self.nmb_scores):
                self.action_index = result[:self.nmb_scores]
                self.scores = result[self.nmb_scores:]
                self.New_out[1] = True
            else:
                self.status = result[:self.nmb_status]
                self.action_index = result[-self.nmb_scores*2:-self.nmb_scores]
                self.scores = result[-self.nmb_scores:]
                self.New_out = [True,True]
            self.count -= 1

    def add(self):
        """
        the counter for receiving resutls 
        """
        with self.cond:
            self.count += 1
    
    def reset(self):
        """
        reseting the thread (counter)
        """
        with self.cond:
            self.count = 0

    def get(self):
        """
        fetching the latest results
        """
        with self.cond:
            New_out = self.New_out 
            count = self.count
            status = self.status
            action_index =self.action_index
            scores = self.scores
            NoActf = self.NoActf
            test = self.test
            self.New_out = [False,False] 
        return count,status,[action_index,scores],NoActf,test,New_out





    def close(self):
        """
        terminating the parallel thread
        """
        self.key_ = False # breaking the while loop of it's still on in the parallel process(run)
        self.join() # waiting for the parallel process to terminate

"""
# A class inherited from the multiprocessing module
# The class is a client receiving frames from the server processing of the received frame is on the main process
# receiving of the frame is on another process 
class client_rcv_frames_process(mp.Process):
    # IP and port of the server as input to the class
    # mean window to calculate FPS and the speed of receiving data
    # Status is the flag for writing on the image the status of connection 
    def __init__(self,ip,port,status=True,w_max=30): 
        self.client = Network.set_client(ip, port) #setting the connection from the client end 
        self.frames = mp.Queue(0) # Setting the Queue object(Shared memory between process) 
        self.key  = mp.Value('b',True) #the flag across process to manage closing from the parallel process smoothly

        # invoking the base class constructor (defining the class as a multiprocessing class
        # inheriting the prvious initializations from the code besides the shared memory frames Queue and key
        mp.Process.__init__(self)
        self.status = status 
        if self.status:  # If the flag is true then initialize the mean object 
            self.m = Segmentation.mean(w_max)

    # The method in the multiprocessing class that's run in another parallel process
    # it's designed to listen and receive data(frames) from the established connection
    # and close the queue and client when it's done receiving data from the connection because of error or an interrupt 
    def run(self):
        try:
            msglen_sum = 0 #the size of total frames received 
            y = time() #Total time of the receiving
            # The loop to receive frames from the connection
            #self.key.value is a flag to be used by the main process to break the loop and terminate the parallel process
            while (self.key.value):
                x = time() # start recording the time of receiving each frame 
                frame_,msglen = Network.recv_frame(self.client) # receiving a frame 
                msglen_sum += msglen # updating the size of the total frames received

                
                #adding the (frame received,the size of the frame and the total time it took to receive it) in the shared memory buffer
                self.frames.put([frame_,msglen,time()-x]) 

            #calculating and printing the average speed of the connection 
            print('The secound process is terminated \n',
                  'The total average Rate is ',msglen_sum/((time-y)*1000),'KB/s')
            self.client.close() #closing the connection if the for loop is broken
            self.frames.close() #declaring that there is no data will be added to the queue from this process 


        #breaking the connection and terminating the process if there is an error , interruption or connection break 
        except ( KeyboardInterrupt,IOError,OSError)as e:
            
            self.frames.close() #declaring that there is no data will be added to the queue from this process 
            print('The secound process is terminated \n',   #calculating and printing the average speed of the connection 
                  'The total average Rate is ',msglen_sum/((time()-y)*1000),'KB/s')
            self.client.close() #closing the connection if the for loop is broken
        return

    #The method is responsible for consuming data from the queue ,decoding it
    # printing status on it and converting it into RGB if desired 
    def get(self,rgb = True):
        frame_ , msglen , spf = self.frames.get(True,timeout=60) #blocking until getting the data with time out of 60 s 
        frame_ = decode_frame(frame_) #decoding the frames 
        
        if self.status: #printing status if desired 
            [msglen_rate,spf] = self.m.mean([msglen,spf])
            add_status(frame_,s1 = 'size :'+str(msglen/1000)+'KB'
                             ,s2 = 'FPS :'+str(1/spf)
                             ,s3= 'Rate :'+str(msglen_rate/(spf*1000))+'KB/s')
        if rgb: #converting the order of color to RGB
            b,g,r  = cv2.split(frame_)                  # get b,g,r
            frame_ = cv2.merge([r,g,b])       	        # switch it to rgb

        return frame_                                   #Returning the frame as an output
    

    # The method is responsible for Quiting the program swiftly with no daemon process
    # the method is to be using in the main process code 
    def close(self):
        self.key.value = False # breaking the while loop of it's still on in the parallel process(run)
        self.frames.close() # declaring there is no frames will be put on the shared queue from the main process
        self.join() # waiting for the parallel process to terminate
        print('The program has been terminated ')
"""
