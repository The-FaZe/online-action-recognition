import socket
from time import time,sleep
import cv2
from struct import unpack,pack
import numpy as np
import multiprocessing as mp

# A module to generate the mean of the input in real time with window == max

class mean():
    def __init__(self,max = 30):
        self.queue = []
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

def add_status(frame_, s1='put your first  text'
                     , s2='put your second text'
                     , s3='put your third  text'):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText1 = (5,10)
    bottomLeftCornerOfText2 = (5,20)
    bottomLeftCornerOfText3 = (5,30)

    fontScale              = 0.3
    fontColor              = (255,255,255)
    lineType               = 1
    cv2.putText(frame_,s1, bottomLeftCornerOfText1, 
    font, fontScale, fontColor, lineType)

    cv2.putText(frame_,s2, bottomLeftCornerOfText2, 
    font, fontScale, fontColor, lineType)

    cv2.putText(frame_,s3, bottomLeftCornerOfText3, 
    font, fontScale, fontColor, lineType)



# A method to set the host of the connection(initialization) but u have to specify the port
# the connection is TCP/IP - the ip of the server is the ip of the host locally
# number of parallel connections to the server is 1
def set_server(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    #specifying socket type is IPV4"socket.AF_INET" and TCP "SOCK_STREAM"
    ip =socket.gethostbyname(socket.gethostname())              # Getting the local ip of the server
    server_address = (ip,port)      # saving ip and port as tuple
    sock.bind(server_address)       # attaching the socket to the pc (code)
    print ('starting up on',server_address[0],':',server_address[1])
    sock.listen(1)     # start waiting for 1 client to connection 
    print('step#1')
    connection, client_address = sock.accept() #accepting that client and hosting a connection with him
    print('client_address is ',client_address[0],client_address[1])
    return connection  #returning the connection object for further use




# A method to set the client part to set up the connection
# (Ip is the ip of the server we want to connection with)
# Port is the port that the server is listening on
def set_client(ip,port):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#specifying socket type is IPV4"socket.AF_INET" and TCP "SOCK_STREAM"
    server_address = (ip,port)# saving ip and port of the server as tuple
    client.connect(server_address)#Trying to connect to a server with specified Ip and port
    print('The connection has been started') # printing when the connection is successful 
    return client #returning the connection object for further use


# A method to receive any kind of hex file knewing its size (msglen)
# the receiving is done on the both end (client and server) (TCP connection is full duplex communication )
# the connection of the receiving end must be assigned and the max buffer len
def recv_msg(connection,msglen,bufferlen):
    try:
        rcvdlen = 0#the total size of the received packets for the file 
        msg = [];# allocating a None list for the packets to concatenate onto

        #Receving Loop for the packets related of the file(msg)
        while(rcvdlen < msglen):
            
            chunk = connection.recv(min(msglen - rcvdlen, bufferlen)) #receiving chunks (Packets) capped at the bufferlen as max
            
            if (len(chunk) is 0) : # resing an error if there is no packets received 
                raise OSError("socket connection broken")
            
            msg.append(chunk) #concatenating the current received chunk to the total previously received chunks 
            rcvdlen += len(chunk)#updating the total received chunk len 
        msg=b''.join(msg)   #joining the total chunks together as bytes 
        return msg #retrning the msg as o/p
    except ( KeyboardInterrupt,IOError,OSError)as e:
        connection.close()
        raise e

# A method to send frame from either end of the communication with specifying the connection object
# the img input is a pure image without any encoding
# the encoding used here is JPEG then getting the size of the encoded image
# Then sending the size of the image in 4 bytes-size-msg then sending the actual encoded img afterwards
def send_frame(connection,img,Quality=90):
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),Quality] # object of the parameters of the encoding (JPEG with 90% quality) 
    _ ,enc_img = cv2.imencode('.jpg',img,encode_param) # encoding the img in JPEG with specified quality 
    buff = len(enc_img) # Getting the len of the encoded image 
    print(buff) #printing the size of the image 
    buff = pack('>L',buff) #converting the size into 4 bytes(struct) length msg ,(L means unsigned long),(> means big endian)
    enc_img1 = enc_img.tostring() #converting the encoded image array into bytes(struct) of the actual memory
    connection.sendall(buff) #sending the size of the frame(img)
    connection.sendall(enc_img1)#sending the actual img 


# A method to recieved a frame from either side of the connection
# connecion is the socket object of the connection from either side(receiving end)
def recv_frame(connection):
    msglen = recv_msg(connection,4,4)# sending the lenght of the enc
    msglen = unpack(">L", msglen)[0]#converting the size again into an unsigned Long 
    frame = recv_msg(connection,msglen,2048)# receiving the complete frame with packet maximum of 2.048 KB and with len of the received length 
    return frame,msglen # returning the frame as bytes and its length


# Converting the frame from bytes(struct) into array then decoding it 
def decode_frame(frame):
    frame = np.frombuffer(frame,dtype='uint8') # converting the frame into an array again  
    frame = cv2.imdecode(frame,1) # decoding the frame into raw img
    return frame #returning the decoded frame

# A class inherited from the multiprocessing module
# The class is a client receiving frames from the server processing of the received frame is on the main process
# receiving of the frame is on another process 
class Frames_rcv(mp.Process):
    # IP and port of the server as input to the class
    # mean window to calculate FPS and the speed of receiving data
    # Status is the flag for writing on the image the status of connection 
    def __init__(self,ip,port,status=True,w_max=30): 
        self.client = set_client(ip, port) #setting the connection from the client end 
        self.frames = mp.Queue(0) # Setting the Queue object(Shared memory between process) 
        self.key  = mp.Value('b',True) #the flag across process to manage exiting from the parallel process smoothly

        # invoking the base class constructor (defining the class as a multiprocessing class
        # inheriting the prvious initializations from the code besides the shared memory frames Queue and key
        mp.Process.__init__(self)
        self.status = status 
        if self.status:  # If the flag is true then initialize the mean object 
            self.m = mean(w_max)

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
                frame_,msglen = recv_frame(self.client) # receiving a frame 
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
    def get_frame(self,rgb = True):
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
    def exit(self):
        self.key.value = False # breaking the while loop of it's still on in the parallel process(run)
        self.frames.close() # declaring there is no frames will be put on the shared queue from the main process
        self.join() # waiting for the parallel process to terminate
        print('The program has been terminated ')

