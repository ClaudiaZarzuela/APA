import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Andrea\Documents\UCM\4\APA\P6\APA\AAKart_Enunciado\ML')

index = 5
kartData = []

for i in range (index):
    kartData.append(pd.read_csv(r'Kart_' + str(i + 6) + r'.csv'))
    
kartData = pd.concat(kartData, ignore_index = True)

print(kartData)