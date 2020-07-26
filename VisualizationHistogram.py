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
histVal  =5

directory = os.path.join(".//TS2/")
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
             rawdata['O? - INSP ']=rawdata['O? - INSP '].dropna()
             rawdata['Pulse ']=rawdata['Pulse '].dropna()
             if 'Pulse ' in rawdata:
                     for x in rawdata['Pulse ']:
                         pulse.append(x)
                         if(x > 120):
                             gcount += 1
                         elif(x>110):
                             mcount +=1
                         elif(x>100):
                             ncount +=1
                         elif (x< 50):
                             lcount +=1
                     lPulse.append(lcount)
                     gPulse.append(gcount) 
                     mPulse.append(mcount)
                     nPulse.append(ncount)
                     gPercentile.append((gcount/len(rawdata['Pulse ']))*100)
                     lPercentile.append((lcount/len(rawdata['Pulse ']))*100)
                     mPercentile.append((mcount/len(rawdata['Pulse ']))*100)
                     nPercentile.append((ncount/len(rawdata['Pulse ']))*100)
                     gcount = 0
                     lcount = 0  
                     mcount = 0
                     ncount = 0
pcount=np.arange(1,84)                   
f= plt.figure(1)
plt.hist(gPercentile,6,range=[0,6],color='red')
plt.xlabel('Percentage of Time for HR > 120 ')
plt.ylabel('Patients')
f.show()

g= plt.figure(2)
plt.hist(lPercentile,8,range=[0,80],color='green')
plt.xlabel('Percentage of Time for HR < 60')
plt.ylabel('Patients')
g.show()

c= plt.figure(3)
plt.hist(mPercentile,8,range=[0,8],color='orange')
plt.xlabel('Percentage of Time for HR > 110')
plt.ylabel('Patients')
c.show()

d= plt.figure(4)
plt.hist(nPercentile,6,range=[0,60],color='blue')
plt.xlabel('Percentage of Time for HR > 100')
plt.ylabel('Patients')
d.show()

lcount=0
gcount=0
lPercentile=[]
gPercentile=[]

directory = os.path.join(".//TS2/")
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
             if 'ABP - MEAN ' in rawdata:
                     rawdata['ABP - MEAN ']=rawdata['ABP - MEAN '].dropna()
                     for y in rawdata['ABP - MEAN ']:
                         pulse.append(y)
                         if(y < 60):
                             lcount +=1
                     lPulse.append(lcount)
                     lPercentile.append((lcount/len(rawdata['ABP - MEAN ']))*100)
                     lcount = 0  
             if 'ABP - SYS ' in rawdata: 
                     rawdata['ABP - SYS ']=rawdata['ABP - SYS '].dropna()
                     for z in rawdata['ABP - SYS ']:
                         pulse.append(z)
                         if(z < 80):
                             gcount +=1
                     gPulse.append(gcount)
                     gPercentile.append((gcount/len(rawdata['ABP - SYS ']))*100)
                     gcount = 0
pcount=np.arange(1,52)
h= plt.figure(5)
plt.hist(gPercentile,7,range=[0,7],color='brown')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for ABP - SYS < 80')
h.show()

pcount=np.arange(1,67)
i= plt.figure(6)
plt.hist(lPercentile,7,range=[0,35],color='purple')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for ABP - Mean < 60')
i.show()

lcount=0
gcount=0
lPercentile=[]
gPercentile=[]

directory = os.path.join(".//TS2/")
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
             if 'PEEP ' in rawdata:
                     rawdata['PEEP ']=rawdata['PEEP '].dropna()
                     for y in rawdata['PEEP ']:
                         pulse.append(y)
                         if(y < 5):
                             lcount +=1
                     lPulse.append(lcount)
                     lPercentile.append((lcount/len(rawdata['PEEP ']))*100)
                     lcount = 0  
             if 'PEEP ' in rawdata: 
                     rawdata['PEEP ']=rawdata['PEEP '].dropna()
                     for z in rawdata['PEEP ']:
                         pulse.append(z)
                         if(z > 10):
                             gcount +=1
                     gPulse.append(gcount)
                     gPercentile.append((gcount/len(rawdata['PEEP ']))*100)
                     gcount = 0
pcount=np.arange(1,39)
j= plt.figure(7)
plt.hist(gPercentile,8,range=[0,1.6],color='gold')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for PEEP > 10')
j.show()

pcount=np.arange(1,39)
k= plt.figure(8)
plt.hist(lPercentile,5,range=[0,100],color='black')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for PEEP < 5')
k.show()

lcount=0
gcount=0
lPercentile=[]
gPercentile=[]


directory = os.path.join(".//TS2/")
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
             if 'Ppeak ' in rawdata:
                     rawdata['Ppeak ']=rawdata['Ppeak '].dropna()
                     for y in rawdata['Ppeak ']:
                         pulse.append(y)
                         if(y > 30):
                             lcount +=1
                     lPulse.append(lcount)
                     lPercentile.append((lcount/len(rawdata['Ppeak ']))*100)
                     lcount = 0  
             if 'O? - INSP ' in rawdata: 
                     rawdata['O? - INSP ']=rawdata['O? - INSP '].dropna()
                     for z in rawdata['O? - INSP ']:
                         pulse.append(z)
                         if(z > 70):
                             gcount +=1
                     gPulse.append(gcount)
                     gPercentile.append((gcount/len(rawdata['O? - INSP ']))*100)
                     gcount = 0
pcount=np.arange(1,39)
l= plt.figure(9)
plt.hist(lPercentile,7,range=[0,70],color='lime')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for Ppeak > 30')
l.show()

pcount=np.arange(1,84)
m= plt.figure(10)
plt.hist(gPercentile,5,range=[0,100],color='pink')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for FiO2 > 70')
m.show()

lcount=0
gcount=0
lPercentile=[]
gPercentile=[]

directory = os.path.join(".//TS2/")
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
             if 'CO? - ET ' in rawdata:
                     rawdata['CO? - ET ']=rawdata['CO? - ET '].dropna()
                     for y in rawdata['CO? - ET ']:
                         pulse.append(y)
                         if(y < 25):
                             lcount +=1
                     lPulse.append(lcount)
                     lPercentile.append((lcount/len(rawdata['CO? - ET ']))*100)
                     lcount = 0  
             if 'CO? - ET ' in rawdata: 
                     rawdata['CO? - ET ']=rawdata['CO? - ET '].dropna()
                     for z in rawdata['CO? - ET ']:
                         pulse.append(z)
                         if(z > 45):
                             gcount +=1
                     gPulse.append(gcount)
                     gPercentile.append((gcount/len(rawdata['CO? - ET ']))*100)
                     gcount = 0
pcount=np.arange(1,84)
m= plt.figure(11)
plt.hist(lPercentile,8,range=[0,40],color='coral')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for CO? - ET < 25')
m.show()

pcount=np.arange(1,84)
n= plt.figure(12)
plt.hist(gPercentile,7,range=[0,70],color='teal')
plt.ylabel('Patients')
plt.xlabel('Percentage of Time for CO? - ET > 45')
n.show()