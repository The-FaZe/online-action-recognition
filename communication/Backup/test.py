from TCP import mean
from time import time
t1=time()
for y in range (1000):
    x =mean(0.1*y)
t2=time()
print((t2-t1)/1000)
