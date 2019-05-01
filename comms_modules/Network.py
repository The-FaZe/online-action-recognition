import socket
from time import time,sleep
import cv2
from struct import unpack,pack
import numpy as np
import multiprocessing as mp
from threading import Thread
import subprocess as sp
from paramiko.client import SSHClient,AutoAddPolicy
from paramiko.ssh_exception import *


def tunneling_cmd_hpc_server(user,path,local_port):
    port = sp.run(["shuf","-i8000-9999","-n1"],capture_output=True)
    port =port.stdout.strip().decode()
    s = "localhost:"+port+":localhost:"+port
    tun_sp=sp.run(["ssh","-R",s,"login01","-N","-f"])
    if path is None:
        s="ssh -L {}:localhost:{} {}@login01.c2.hpc.bibalex.org -N".format(local_port,port,user)
    else:
        s="ssh -L {}:localhost:{} {}@login01.c2.hpc.bibalex.org -N -i {}".format(local_port,port,user,path)
    print("copy the following command \n",s)
    return int(port),tun_sp



def set_server(ip,port,n_conn,Tunnel,hostname=None,username=None):
    connection = []
    transport = None
    if Tunnel:
        
        sshclient = SSHClient()
        
        sshclient.load_system_host_keys()
        
        sshclient.set_missing_host_key_policy(AutoAddPolicy())
        
        try:
            sshclient.connect(hostname=hostname,username=username)
        except(BadHostKeyException,SSHException
            ,AuthenticationException) as e:
            print("problem hapened ssh into {}".format(hostname))
            raise e
            sshclient.get_transport().close()
            return None,None
        try:
            try:
                transport = sshclient.get_transport()
                port=transport.request_port_forward(address=ip,port=port)
            except(SSHException):
                print("the Node refused the Given port {}".format(port))
                port = 0
                port=transport.request_port_forward(address=ip,port=port)
            print("port {}".format(port))
            print ('starting up on {} : {}'.format(ip,port))
            for i in range(n_conn):
                connection.append(transport.accept(None))
                print('client_address is {}:{} '.format(*connection[i].getpeername()))
        except(BadHostKeyException,SSHException
            ,AuthenticationException)as e:
            print("an unexpected problem has been faced in request_port_forward")
            for i in connection:
                i.close()
            transport.close()
            return None ,None


        #port,s=tunneling_cmd_hpc_server(user=user,path=path,local_port=port) 
    else:
        #s = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    #specifying socket type is IPV4"socket.AF_INET" and TCP "SOCK_STREAM"
        server_address = (ip,port)      # saving ip and port as tuple
        try:
            sock.bind(server_address)       # attaching the socket to the pc (code)
            print ('starting up on',server_address[0],':',server_address[1])
            sock.listen(n_conn)             # start waiting for 1 client to connection 
            for i in range(n_conn):
                connection_, client_address = sock.accept() #accepting that client and hosting a connection with him
                print('client_address is {}:{} '.format(*client_address))
                connection.append(connection_)
        except(OSError) as e:
            print("Making a server Failed")
            raise e
            for i in connection:
                i.close()
            return None,None


    print('The number of connections started successfully = {}'.format(n_conn)) # printing when the connection is successful 
    return connection,transport  #returning the connection object for further use





# A method to set the client part to set up the connection
# (Ip is the ip of the server we want to connection with)
# Port is the port that the server is listening on
def set_client(ip,port,numb_conn,Tunnel,hostname=None,username=None,Key_path=None,passphrase=None):
    client = []
    transport = None
    if Tunnel:
        server_address = (ip,port)
        sshclient = SSHClient()
        sshclient.load_system_host_keys()
        sshclient.set_missing_host_key_policy(AutoAddPolicy())
        try:
            sshclient.connect(hostname=hostname,username=username,passphrase=passphrase
                ,key_filename=Key_path)
            transport = sshclient.get_transport()
        except(BadHostKeyException,SSHException
            ,AuthenticationException) as e:
            print("A problem trying to connection to the Host {}".format(hostname))
            raise e
            transport.close()
            return None,None
        try:
            for i in range(numb_conn):
                client.append(transport.open_channel(kind='direct-tcpip',src_addr=server_address,dest_addr=server_address))
                client[i].setblocking(True)
        except SSHException as e:
            print("problem Has been faced trying to make direct-tcpip conneciton")
            raise e
            for i in client:
                i.close()
            sshclient.get_transport().close()
            return None,None
        except(KeyboardInterrupt):
            for i in client:
                i.close()
            sshclient.get_transport().close()
            return None,None
    else:
        server_address = (ip,port)
        try :
            for i in range(numb_conn):
                client.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                client[i].setblocking(True)
                client[i].connect(server_address)
        except(OSError):
            print("The connection #{} failed".format(i+1))
            print("The process is stopping")
            return None,None
        except(KeyboardInterrupt):
            for i in client:
                i.close()
            sshclient.get_transport().close()
            return None,None

    return client,transport #returning the connection object for further use


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
def send_frame(connection,img,Quality=90,active_reset=False):
    if active_reset :
        buff = pack('>L',0)
        connection.sendall(buff)
        confirmation=recv_msg(connection,1,1)
        if confirmation != b'\xff':
            print("the received confirmation is not right")
            raise OSError

    else:
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),Quality] # object of the parameters of the encoding (JPEG with 90% quality) 
        _ ,enc_img = cv2.imencode('.jpg',img,encode_param) # encoding the img in JPEG with specified quality 
        buff = len(enc_img) # Getting the len of the encoded image  
        if buff is 0:
            print("the frame is empty") 
            raise OSError
        buff = pack('>L',buff) #converting the size into 4 bytes(struct) length msg ,(L means unsigned long),(> means big endian)
        enc_img1 = enc_img.tostring() #converting the encoded image array into bytes(struct) of the actual memory
        connection.sendall(buff) #sending the size of the frame(img)
        connection.sendall(enc_img1)#sending the actual img 


# A method to recieved a frame from either side of the connection
# connecion is the socket object of the connection from either side(receiving end)
def recv_frame(connection):
    msglen = recv_msg(connection,4,4) # sending the lenght of the enc
    msglen = unpack(">L", msglen)[0] #converting the size again into an unsigned Long 

    if msglen is 0:
        frame = None
    else:
        frame = recv_msg(connection,msglen,2048) # receiving the complete frame with packet maximum of 2.048 KB and with len of the received length 


    return frame,msglen # returning the frame as bytes and its length


# Converting the frame from bytes(struct) into array then decoding it 
def decode_frame(frame):
    frame = np.frombuffer(frame,dtype='uint8') # converting the frame into an array again  
    frame = cv2.imdecode(frame,1) # decoding the frame into raw img
    return frame #returning the decoded frame
