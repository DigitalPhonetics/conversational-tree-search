import torch
import time

a = torch.zeros(60000,100000,device='cuda:0')
b = torch.zeros(40000,100000,device='cuda:0')

while True:
    time.sleep(3600)
    print("ALIVE")

