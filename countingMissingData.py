# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:58:20 2019

@author: Atif Ul Aftab
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
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
                   patient_no.append(comp[0])
           if not found:
             print('not found')
           else :
             rawdata=pd.read_csv(".//TS//"+file,delimiter=';')
             if 'Date' in rawdata:
                     for x in rawdata['Date']:
                         gcount += 1
                     gPulse.append(gcount) 
                     gcount = 0
             if 'Pulse ' in rawdata:
                     for x in rawdata['Pulse ']:
                         if math.isnan(x):
                             lcount +=1
                     lPulse.append(lcount)
                     print(lcount)
                     lcount = 0
             if 'CO? - ET ' in rawdata:
                     for x in rawdata['CO? - ET ']:
                         if math.isnan(x):
                             mcount +=1
                     mPulse.append(lcount)
                     mcount = 0
             if 'O? - INSP ' in rawdata:
                     for x in rawdata['O? - INSP ']:
                         if math.isnan(x):
                             ncount +=1
                     nPulse.append(lcount)
                     ncount = 0
dataset = []
height=len(gPulse)

gPulse = map(str,gPulse)
gPulse = list(gPulse)

lPulse = map(str,lPulse)
lPulse = list(lPulse)

nPulse = map(str,nPulse)
nPulse = list(nPulse)

mPulse = map(str,mPulse)
mPulse = list(mPulse)


for i in range(height-1):
    dataset.append(patient_no[i]+','+gPulse[i]+','+lPulse[i]+','+nPulse[i]+','+mPulse[i])


with open("MissingCount2.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    for val in dataset:
        writer.writerow([val])
f.close()
print("======================================")
print("\n Missing Count Successfully Completed")