import socket
from time import sleep
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip =socket.gethostbyname(socket.gethostname())
server_address = ("",8345)
client.connect(server_address)
message = 'Hi from the client'
message = message.encode('utf-8')
client.send(message)
data = client.recv(1024).decode('utf-8')
print (data)
client.close()
