# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:58:20 2019

@author: Atif Ul Aftab
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
                     gPulse.append(gcount) 
                     gcount = 0
pcount=np.arange(1,84)                   
f= plt.figure(1)
plt.bar(pcount,gPulse,color='blue')
plt.xlabel('Patients')
plt.ylabel('Surgery Duration (Seconds)')
f.show()