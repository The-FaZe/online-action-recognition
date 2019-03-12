from threading import Thread
import socket
import subprocess as sp
def ssh_tun_cmd(port):
	s = "localhost:"+port+":localhost:"+port
	s=sp.run(["ssh","-R",s,"login01","-N"])

port = sp.run(["shuf","-i8000-9999","-n1"],capture_output=True)
port =port.stdout.strip().decode()
T =Thread(target=ssh_tun_cmd,args=(port,))
T.start()
s="ssh -L localhost:"+port+":localhost:"+port+" alex039u4@hpc.bibalex.org -N "
print("copy the following command \n",s)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip =socket.gethostbyname(socket.gethostname())
port = int(port)
server_address = ("localhost",port)
sock.bind(server_address)

sock.listen(1)
print('step#1')
connection, client_address = sock.accept()
print('client_address is ',client_address[0],client_address[1])
data = connection.recv(1024).decode('utf-8')
message = 'Have a nice day'
message = message.encode('utf-8')
print (data)
connection.sendall(message)
connection.close()