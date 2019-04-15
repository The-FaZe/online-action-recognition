import socket
from time import time,sleep
import cv2
from struct import unpack,pack
import numpy as np
import multiprocessing as mp
from threading import Thread
import subprocess as sp


# A method to perfrom tunneling in the cmd on the HPC end
def tunneling_cmd_hpc_server(port):
    s = "localhost:"+port+":localhost:"+port
    s=sp.run(["ssh","-R",s,"login01","-N"])

def tunneling_cmd_hpc_client(port):
    port = str(port)
    s = "localhost:"+port+":localhost:"+port
    s = sp.run(["ssh","-L",s,"alex039u4@hpc.bibalex.org","-N"])

# A method to perform the tunneling and pick the port to work on .
def ssh_tun():
    port = sp.run(["shuf","-i8000-9999","-n1"],capture_output=True)
    port =port.stdout.strip().decode()
    T =Thread(target=tunneling_cmd_hpc_server,args=(port,))
    T.start()
    print("port:",port)
    s="ssh -L localhost:"+port+":localhost:"+port+" alex039u4@hpc.bibalex.org -N "
    print("copy the following command \n",s)
    return int(port),T



# A method to set the host of the connection(initialization) but u have to specify the port if the tunneling isn't used
# the connection is TCP/IP - the ip of the server is the ip of the host locally
# n is the number of parallel connections to the server is (default is 1)
def set_server(port=None,Tunnel=True,n=2):
    if Tunnel:
        port,T = ssh_tun()
    else:
        T = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    #specifying socket type is IPV4"socket.AF_INET" and TCP "SOCK_STREAM"
    if Tunnel:
        ip = "localhost"
    else:
        ip =socket.gethostbyname(socket.gethostname())              # Getting the local ip of the server
    server_address = (ip,port)      # saving ip and port as tuple
    sock.bind(server_address)       # attaching the socket to the pc (code)
    print ('starting up on',server_address[0],':',server_address[1])
    sock.listen(n)     # start waiting for 1 client to connection 
    print('step#1')
    connection, client_address = sock.accept() #accepting that client and hosting a connection with him
    print('client_address is ',client_address[0],client_address[1])
    connection2, client_address = sock.accept() #accepting that client and hosting a connection with him
    print('client_address is ',client_address[0],client_address[1])
    connection = (connection,connection2)
    return connection,T  #returning the connection object for further use




# A method to set the client part to set up the connection
# (Ip is the ip of the server we want to connection with)
# Port is the port that the server is listening on
def set_client(port,ip="",Tunnel=True):
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
