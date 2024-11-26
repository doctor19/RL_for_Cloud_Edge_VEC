import random as rd
import numpy as np
from pathlib import Path
import os
from config import DATA_LOCATION
path =os.path.abspath(__file__)
path =Path(path).parent.parent
for i in range(200):
    with open("{}/{}/datatask{}.csv".format(str(path),DATA_LOCATION,i),"w") as output:
        # indexs=rd.randint(900,1200)
        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        # m1 = np.random.randint(1000,2000,indexs) # p in
        # m2 = np.random.randint(100,200,indexs) # p out
        # m3 = np.random.randint(500,1500,indexs) #Computational resource
        # m4 = 1+np.random.rand(indexs)*2 #deadline
        
        # indexs=1200
        # seconds = 60
        # m = np.sort(np.random.randint(i*10*seconds,(i+1)*10*seconds,indexs)/10)
        # m1 = np.random.randint(2000,3000,indexs)/1000 # p in Mb
        # m2 = np.random.randint(15,20,indexs)/1000 # p out Mb
        # m3 = np.random.randint(500,800,indexs)/1000 #Computational resource Gigacycle
        # m4 = 0.5+np.random.rand(indexs)/2 #deadline
        
        indexs=3500
        seconds = 60
        # To make the diff as 0.1s
        m = np.sort(np.random.randint(i*10*seconds,(i+1)*10*seconds,indexs)/10)
        m1 = np.random.randint(1500,2000,indexs)/1000 # p in Mb
        m2 = np.random.randint(100,150,indexs)/1000 # p out Mb
        m3 = np.random.randint(300,500,indexs)/1000 #Computational resource Gigacycle
        m4 = 0.5+np.random.rand(indexs)/2 #deadline
        
        
        # indexs=rd.randint(1000,1000)
        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        # m1 = np.random.randint(1000,1200,indexs)
        # m2 = np.random.randint(100,110,indexs)
        # m3 = np.random.randint(2,3,indexs)
        # m4 = 5+np.random.rand(indexs)*2
        
        for j in range(indexs):
            output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))
    #import pdb;pdb.set_trace()