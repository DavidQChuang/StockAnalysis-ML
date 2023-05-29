import time
import random
import sys
import math

# stdout = sys.stdout

# sys.stdout = 

epochs = 12
loss = 0.01
val_loss = 0.005
    
for i in range(1, epochs):
    time.sleep(0.1)
    print(f'Epoch {i}/{epochs}')
        
    start = time.time()
    iters = 75
    for l in range(iters):
        now = time.time()
        step = random.randint(10, 20)
        loss *= 0.999 - 0.001 * epochs / i
        val_loss *= 0.999 - 0.001 * epochs / i
        
        time.sleep(step/1000)
        
        progress = round(30 * l / iters)
        prog_bar = '='*progress + '-'*(30-progress)
            
        print('%d/%d [%s] - %ds %dms/step - loss: %.4f - val_loss: %.4f'%(
            l, iters, prog_bar, math.floor(now-start), step, loss, val_loss), end="\r")
    print() 
    
    
print("\n")

print("Loss: ")
for i in range(10):
    time.sleep(0.01)
    print('Downloading File FooFile.txt [%d%%]'%i, end="\r")
    
print("\n")