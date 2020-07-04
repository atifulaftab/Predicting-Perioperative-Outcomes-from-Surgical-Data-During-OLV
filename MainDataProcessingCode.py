# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:18:20 2019

@author: atifu
"""

import os
import pandas as pd
import csv


heartRateMean = []
heartRateMax = []
peepMean = []
peepMax = []
ppeekMean = []
ppeekMax = []
tidalVolumeMean = []
tidalVolumeMax = []
o2InsMean = []
o2InsMax = []
complication = []
delayed = []
surgeryTime = []
cardiac = []
fileName = []
patientId =[]
notfound = []
FEV1 = []
DLCO = []
FVC = []
FEV1_FVC = []
smoking = []
otype = []
co = []
o_time = []
i=0
delayed2 = []
delayed3 = []




pulse1 = []
pulse2 = []
pulse3 = []
pulse4 = []

count1 = 0
count2 = 0
count3 = 0
count4 = 0

abpMean = []
abpSys = []

peep1 = []
peep2 = []

pPeak = []
oIns = []

cMin = []
cMax = []

vt = []

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
                   print(comp[0])
                   complication.append(comp[2])
                   delayed.append(comp[3])
                   patientId.append(comp[0])
                   FEV1.append(comp[5])
                   DLCO.append(comp[6])
                   FVC.append(comp[7])
                   FEV1_FVC.append(comp[8])
                   smoking.append(comp[9])
                   otype.append(comp[10])
                   co.append(comp[11])
                   o_time.append(comp[12])
                   delayed2.append(comp[13])
                   delayed3.append(comp[14])
                   found = True
           if not found:
             print('not found')
           else :
             rawdata=pd.read_csv(".//TS//"+file,delimiter=';')
             rawdata['O? - INSP ']=rawdata['O? - INSP '].dropna()
             rawdata['Pulse ']=rawdata['Pulse '].dropna()
             if 'Pulse ' in rawdata:
                     length=len(rawdata['Pulse '])
                     mean=rawdata[0:length]['Pulse '].mean()
                     maxi=rawdata[0:length]['Pulse '].max()
                     heartRateMean.append(str(mean))
                     heartRateMax.append(str(maxi))
             if 'O? - INSP ' in rawdata:
                     length=len(rawdata['O? - INSP '])
                     mean=rawdata[0:length]['O? - INSP '].mean()
                     maxi=rawdata[0:length]['O? - INSP '].max()
                     o2InsMean.append(str(mean))
                     o2InsMax.append(str(maxi))
             if 'Pulse ' in rawdata:
                     for x in rawdata['Pulse ']:
                         if(x > 120):
                             count1 += 1
                         elif(x>110):
                             count2 +=1
                         elif(x>100):
                             count3 +=1
                         elif (x< 50):
                             count4 +=1
                     pulse1.append((count1/len(rawdata['Pulse '])))
                     pulse2.append((count2/len(rawdata['Pulse '])))
                     pulse3.append((count3/len(rawdata['Pulse '])))
                     pulse4.append((count4/len(rawdata['Pulse '])))
                     count1 = 0
                     count2 = 0
                     count3 = 0
                     count4 = 0
             else:
                 pulse1.append(-1)
                 pulse2.append(-1)
                 pulse3.append(-1)
                 pulse4.append(-1)
             if 'ABP - MEAN ' in rawdata:
                     rawdata['ABP - MEAN ']=rawdata['ABP - MEAN '].dropna()
                     for y in rawdata['ABP - MEAN ']:
                         if(y < 60):
                             count1 +=1
                     abpMean.append((count1/len(rawdata['ABP - MEAN '])))
                     count1 = 0
             else:
                 abpMean.append(-1)
             if 'ABP - SYS ' in rawdata: 
                     rawdata['ABP - SYS ']=rawdata['ABP - SYS '].dropna()
                     for z in rawdata['ABP - SYS ']:
                         if(z < 80):
                             count1 +=1
                     abpSys.append((count1/len(rawdata['ABP - SYS '])))
                     count1 = 0 
             else:
                 abpSys.append(-1)
             if 'PEEP ' in rawdata:
                     rawdata['PEEP ']=rawdata['PEEP '].dropna()
                     for y in rawdata['PEEP ']:
                         if(y < 5):
                             count1 +=1
                     peep1.append((count1/len(rawdata['PEEP '])))
                     count1 = 0
             else:
                 peep1.append(-1)
             if 'PEEP ' in rawdata: 
                     rawdata['PEEP ']=rawdata['PEEP '].dropna()
                     for z in rawdata['PEEP ']:
                         if(z > 10):
                             count1 +=1
                     peep2.append((count1/len(rawdata['PEEP '])))
                     count1 = 0
             else:
                 peep2.append(-1)
             if 'Ppeak ' in rawdata:
                     rawdata['Ppeak ']=rawdata['Ppeak '].dropna()
                     for y in rawdata['Ppeak ']:
                         if(y > 30):
                             count1 +=1
                     pPeak.append((count1/len(rawdata['Ppeak '])))
                     count1 = 0
             else:
                 pPeak.append(-1)
             if 'O? - INSP ' in rawdata: 
                     rawdata['O? - INSP ']=rawdata['O? - INSP '].dropna()
                     for z in rawdata['O? - INSP ']:
                         if(z > 70):
                             count1 +=1
                     oIns.append((count1/len(rawdata['O? - INSP '])))
                     count1 = 0
             else:
                 oIns.append(-1)
             if 'CO? - ET ' in rawdata:
                     rawdata['CO? - ET ']=rawdata['CO? - ET '].dropna()
                     for y in rawdata['CO? - ET ']:
                         if(y < 25):
                             count1 +=1
                     cMin.append((count1/len(rawdata['CO? - ET ']))*100)
                     count1 = 0 
             else:
                 cMin.append(-1)
             if 'CO? - ET ' in rawdata: 
                     rawdata['CO? - ET ']=rawdata['CO? - ET '].dropna()
                     for z in rawdata['CO? - ET ']:
                         if(z > 45):
                             count1 +=1
                     cMax.append((count1/len(rawdata['CO? - ET ']))*100)
                     count1 = 0
             else:
                 cMax.append(-1)
             if 'TV ' in rawdata:
                 vt.append(-100)
             else:
                 vt.append(-1)
                 
dataset = []

height=len(patientId)

pulse1 = map(str,pulse1)
pulse1 = list(pulse1)

pulse2 = map(str,pulse2)
pulse2 = list(pulse2)

pulse3 = map(str,pulse3)
pulse3 = list(pulse3)

pulse4 = map(str,pulse4)
pulse4 = list(pulse4)

abpMean = map(str,abpMean)
abpMean = list(abpMean)

abpSys = map(str,abpSys)
abpSys = list(abpSys)

peep1 = map(str,peep1)
peep1 = list(peep1)

peep2 = map(str,peep2)
peep2 = list(peep2)

pPeak = map(str,pPeak)
pPeak = list(pPeak)

oIns = map(str,oIns)
oIns = list(oIns)

cMin = map(str,cMin)
cMin = list(cMin)

cMax = map(str,cMax)
cMax = list(cMax)

vt = map(str,vt)
vt = list(vt)
for i in range(height-1):
    dataset.append(patientId[i]+','+heartRateMean[i]+','+heartRateMax[i]+','+o2InsMean[i]+','+o2InsMax[i]+','+pulse1[i]+','+pulse2[i]+','+pulse3[i]+','+pulse4[i]+','+abpMean[i]+','+abpSys[i]+','+peep1[i]+','+peep2[i]+','+pPeak[i]+','+oIns[i]+','+cMin[i]+','+cMax[i]+','+o_time[i]+','+FEV1[i]+','+DLCO[i]+','+co[i]+','+otype[i]+','+complication[i]+','+delayed[i]+','+delayed2[i]+','+delayed3[i])


with open("abc.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    for val in dataset:
        writer.writerow([val])
f.close()
print("======================================")
print("\n Data Processng Successfully Completed")
