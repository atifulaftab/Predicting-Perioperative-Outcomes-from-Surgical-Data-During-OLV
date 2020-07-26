# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:48:27 2019

@author: atifu
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

patient_no = []
pulse = []
gPulse = []
lPulse = []
mPulse = []
nPulse = []
gcount = 0
lcount = 0
mcount = 0
ncount = 0
mPercentile = []
gPercentile = []
lPercentile = []
nPercentile = []
lengthCount = 0
totalLength = 0
directory = os.path.join(".//TS/")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           print(file)
           myFile= open("LabelData.csv")
           found = False
           for comp in myFile:
               comp=comp.strip().split(',')
               if(comp[1]==file):                   
                   found = True                   
           if not found:
             print('not found')
           else :
             rawdata=pd.read_csv(".//TS//"+file,delimiter=';')
             if 'Date' in rawdata:
                     for x in rawdata['Date']:
                         gcount += 1
                     if gcount>487:    
                         gPulse.append(gcount)
                         totalLength= totalLength + gcount
                         print(gcount)
                     gcount = 0
# pcount=np.arange(0,81)                   
# f= plt.figure(1)
# plt.bar(pcount,gPulse)
# plt.xlabel('Patients')
# plt.ylabel('Surgery Duration')
# f.show()
print("Mean Length: ",totalLength/80)
print("Mean Length(Minutes): ",(totalLength/80)/60)
pcount=np.arange(0,81)
m= plt.figure(11)
plt.hist(gPulse,25,range=[0,25000],color='blue')
plt.ylabel('Patients')
plt.xlabel('Surgery Duration (Seconds)')
m.show()