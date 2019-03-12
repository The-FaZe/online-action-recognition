import socket
from time import sleep


print("start python code........")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip =socket.gethostbyname(socket.gethostname())
port = 6666
server_address = (ip,port)
print("ip: ",ip)
sock.bind(server_address)
print ('starting up on',server_address[0],':',server_address[1])
sock.listen(1)
print('step#1')
connection, client_address = sock.accept()
print('client_address is ',client_address[0],client_address[1])
data = connection.recv(1024).decode('utf-8')
message = 'Have a nice day'
message = message.encode('utf-8')
print (data)
connection.sendall(message)
sleep(10)
connection.close()
