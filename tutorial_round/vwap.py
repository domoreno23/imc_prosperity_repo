import pandas as pd


def Vwap(priceData):
    price = priceData[0]
    volume = priceData[1]
    cumulativeVolume = priceData[2]
    vwap = priceData[3]
    n = price * volume

    cuN = priceData[4]
    cuN +=n

    cumulativeVolume += volume
    vwap = cuN / cumulativeVolume
    return cumulativeVolume, vwap, cuN


x = [5,10,0,0]
y = [10, 10, ]


t1 = [10,10]
t2 = [9,5]
t3 = [8,20]
t4 = [9,1]
t5 = [10,2]


trades = [t1,t2,t3,t4,t5]

cuVol = 0
vwap = 0
cuN = 0 

for i in trades:
    i.append(cuVol)
    i.append(vwap)
    i.append(cuN)


    cuVol, vwap, cuN = Vwap(i)
    #print('cuVol:  ', cuVol)
    print('vwap: ', vwap)
    #print('cuN: ', cuN)




    
        



