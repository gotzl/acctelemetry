import os
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

from ldparser import ldparser


def laps(f):
    laps = []
    tree = ET.parse(os.path.splitext(f)[0]+".ldx")
    root = tree.getroot()

    # read lap times
    for lap in root[0][0][0][0]:
        laps.append(float(lap.attrib['Time'])*1e-6)
    return laps


def laps_limits(laps, freq, n):
    """find the start/end indizes of the data for each lap
    """
    laps_limits = list([0])
    laps_limits.extend((np.array(laps)*freq).astype(int))
    laps_limits.extend([n])
    return list(zip(laps_limits[:-1], laps_limits[1:]))


def createDataFrame(channs, laps_limits):
    # convert some of the data from ld file to integer
    channs[7].data = list(map(int, channs[7].data))
    channs[11].data = list(map(int, channs[11].data))
    channs[12].data = list(map(int, channs[12].data))

    # create a data frame with the data from the ld file
    df = pd.DataFrame({i.name.lower(): i.data for i in channs[1:]})

    # create list with the total distance
    s = []
    for spd in df.speed:
        if len(s)==0:
            s_, ds_ = 0, 0
        else:
            s_, ds_ = s[-1], spd/channs[4].freq
        s.append(s_+ds_)

    # create list with total time
    t = np.arange(len(channs[4].data))*(1/channs[4].freq)


    # create list with the lap number, distance in lap, time in lap
    s = np.array(s)
    l, sl, tl = [],[],[]
    for n, (n1,n2) in enumerate(laps_limits):
        l.extend([n]*(n2-n1))
        sl.extend(list(s[n1:n2]-s[n1]))
        tl.extend(list(t[n1:n2]-t[n1]))

    # calculate oversteer, based on math in ACC MoTec workspace
    wheelbase = 2.645
    a= np.sign(df.g_lat) * (wheelbase * df.g_lat / (df.speed*df.speed))
    b= np.sign( ((df.steerangle/11) * (np.pi/180) * df.g_lat).mean() ) * ((df.steerangle/11) * (np.pi/180))
    oversteer = (a-b)*(180/np.pi)


    # add the lists to the dataframe
    return pd.concat([df, pd.DataFrame(
        {'lap':l,
         'g_sum': df.g_lon.abs()+df.g_lat.abs(),
         'speedkmh':df.speed*3.6,
         'oversteer':oversteer,
         'dist':s,'dist_lap':sl,
         'time':t,'time_lap':tl})], axis=1)


def scanFiles(files):
    data = []
    for f in files:
        if not os.path.isfile(os.path.splitext(f)[0]+".ldx"): continue
        head = ldparser.ldhead(f)
        laps_ = laps(f)
        for i, lap in enumerate(laps_):
            if i>0: lap -= laps_[i-1]
            data.append((os.path.basename(f),
                         head.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                         head.descr1, head.descr2, i,
                         "%i:%02i.%03i"%(lap//60,lap%60,(lap*1e3)%1000),
                        ))

    if len(data)==0:
        return dict()

    data = np.array(data)
    return dict(
        name=data[:,0],
        datetime=data[:,1],
        track=data[:,2],
        car=data[:,3],
        lap=data[:,4],
        time=data[:,5],
    )



