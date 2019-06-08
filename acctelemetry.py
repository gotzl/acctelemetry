import os, glob, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm

import xml.etree.ElementTree as ET

from ldparser import ldparser

norm = mplcolors.Normalize(vmin=-.1, vmax=1)
cmapg = mplcm.ScalarMappable(norm=norm, cmap=mplcm.Greens)
cmapr = mplcm.ScalarMappable(norm=norm, cmap=mplcm.Reds)
cmapb = mplcm.ScalarMappable(norm=norm, cmap=mplcm.Blues)


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


def createDataFrame(file_, channs, laps_times, laps_limits):
    # convert some of the data from ld file to integer
    channs[7].data = list(map(int, channs[7].data))
    channs[11].data = list(map(int, channs[11].data))
    channs[12].data = list(map(int, channs[12].data))

    # create a data frame with the data from the ld file
    df = pd.DataFrame({i.name.lower(): i.data for i in channs[1:]})

    # create list with the total distance
    ds = (df.speed / channs[4].freq)
    s = ds.cumsum()

    # create list with total time
    t = np.arange(len(channs[4].data))*(1/channs[4].freq)

    # create list with the lap number, distance in lap, time in lap
    s = np.array(s)
    l, sl, tl = [],[],[]
    for n, (n1,n2) in enumerate(laps_limits):
        l.extend([n]*(n2-n1))
        sl.extend(list(s[n1:n2]-s[n1]))
        tl.extend(list(t[n1:n2]-t[n1]))

    # calculate x/y position on track from speed and g_lat
    gN = 9.81
    r = 1 / (gN * df.g_lat/df.speed.pow(2))
    alpha = ds / r

    heading = alpha.cumsum()

    # dx = (2*r*np.tan(alpha/2)) * np.cos(heading)
    # dy = (2*r*np.tan(alpha/2)) * np.sin(heading)
    dx = ds * np.cos(heading)
    dy = ds * np.sin(heading)

    # add the lists to the dataframe
    df = pd.concat([df, pd.DataFrame(
        {'file':file_,'lap':l,
         'g_sum': df.g_lon.abs()+df.g_lat.abs(),
         'speedkmh':df.speed*3.6,
         'alpha':alpha, 'heading':heading,
         'dx':dx, 'dy':dy,'ds':ds,
         'dist':s,'dist_lap':sl,
         'time':t,'time_lap':tl})], axis=1)

    # calculate correction to close the track
    # use best lap
    fastest = np.argmin(laps_times)
    df_ = df[(df.lap==fastest+1)]
    fac = 1.
    dist = None
    while True:
        dx = df_.ds * np.cos(df_.heading*fac)
        dy = df_.ds * np.sin(df_.heading*fac)
        end = (dx.cumsum()).values[-1],(dy.cumsum()).values[-1]
        # print(end, dist, fac)

        newdist = np.sqrt(end[0]**2+end[1]**2)
        if dist is not None and newdist>dist: break
        dist = newdist
        fac -= 0.0001

    # recalculate with correction
    df.alpha = df.alpha*fac
    df.heading = alpha.cumsum()
    df.dx = ds * np.cos(heading*fac)
    df.dy = ds * np.sin(heading*fac)
    x = df.dx.cumsum()
    y = df.dy.cumsum()
    # xl,yl = [],[]
    # for n, (n1,n2) in enumerate(laps_limits):
    #     xl.extend(list(x[n1:n2]-x[n1]))
    #     yl.extend(list(y[n1:n2]-y[n1]))

    # calculate oversteer, based on math in ACC MoTec workspace
    wheelbase = 2.645
    neutral_steering = (wheelbase * alpha * 180/np.pi).rolling(10).mean()

    steering_corr= (df.steerangle/11)
    oversteer  = np.sign(df.g_lat) * (neutral_steering-steering_corr)
    understeer = oversteer.copy()

    indices = oversteer < 0
    oversteer[indices] = 0

    indices = understeer > 0
    understeer[indices] = 0


    # add the lists to the dataframe
    df = pd.concat([df, pd.DataFrame(
        {'x':x,'y':y,
         # 'xl':xl,'yl':yl,
         'steering_corr':steering_corr,
         'oversteer':oversteer,
         'understeer':understeer})], axis=1)
    return df


def lapdelta(df, reference, target):
    """
    # returns delta times against reference for target lap
    # - win of target vs reference -> green
    # - loss of target vs reference -> red
    :param df:          dataframe with the laps
    :param reference:   lap number of the reference lap
    :param target:      lap number of the lap to compare
    :return:            (list of delta times), df[lap==reference]
    """

    a,b = target, reference
    df_a = df[(df.file==a[0]) & (df.lap==a[1])]
    df_b = df[(df.file==b[0]) & (df.lap==b[1])]

    def findidx(dist, df, offset=0, direction=1):
        """
        get the idx of df.dist_lap closest to dist
        :param dist:    reference dist
        :param df:      the dataframe to iterate
        :param offset:  offset to checking df
        :param direction: iteration direction
        :return: idx in df
        """
        idx = offset
        if idx>len(df): return len(df)-1

        while idx < len(df) - 1 and idx>=0:
            d = dist - df.dist_lap.values[idx]
            if d<=0 and direction>0: return idx
            if d>=0 and direction<0: return idx
            idx+= direction
        return idx

    # for each track position in a with time ta
    # - find track position in b, interpolate
    dt_, speed, speedkmh, throttle, brake, g_lon, xr, yr = [],[],[],[],[],[],[],[]
    a_idx,b_idx = 0,0
    for idx in range(len(df_a)):
        # the b_idx closest to current track position in a
        b_idx = findidx(df_a.dist_lap.values[idx], df_b, b_idx)

        # the a_idx closest to track position in b at the time of t_a
        # this is needed to get the x,y coords of the reference
        a_idx = findidx(df_b.dist_lap.values[idx if idx<len(df_b) else len(df_b)-1], df_a, a_idx)

        # distance difference of pos in a and b
        ds = df_a.dist_lap.values[idx]-df_b.dist_lap.values[b_idx]
        dt = ds/df_b.speed.values[b_idx]

        # time difference between a and b for current track position in a
        dt_.append(df_a.time_lap.values[idx] - (df_b.time_lap.values[b_idx]+dt))
        xr.append(df_a.x.values[a_idx])
        yr.append(df_a.y.values[a_idx])
        for i in ['speed', 'speedkmh', 'throttle', 'brake', 'g_lon']:
            eval(i).append(df_b[i].values[b_idx])

    df_a = df_a.assign(dt=pd.Series(dt_).values)
    df_a = df_a.assign(xr=pd.Series(xr).values)
    df_a = df_a.assign(yr=pd.Series(yr).values)
    for i in ['speed', 'speedkmh', 'throttle', 'brake', 'g_lon']:
        df_a = df_a.assign(**{'%s_r'%i:pd.Series(eval(i)).values})

    return df_a, df_b


def running_mean(x, N):
    # np.convolve(dt_, np.ones((10,))/10, mode='valid')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def deltacolors(dt, style=None):
    """
    Get colors for delta time
    :param dt:      the delta times
    :param style:   None: color maps to total dt, green -> faster, red -> slower
                    grad: color maps to the derivative of dt, red -> loosing, green -> gaining
    :return:        list of colors
    """
    norm = mplcolors.Normalize(vmin=-.3, vmax=1)
    cmapg = mplcm.ScalarMappable(norm=norm, cmap=mplcm.Greens)
    cmapr = mplcm.ScalarMappable(norm=norm, cmap=mplcm.Reds)
    cmapb = mplcm.ScalarMappable(norm=norm, cmap=mplcm.Blues)

    max_dt = max(abs(dt))
    if style == 'grad':
        dt_grad = np.gradient(list(dt[:19])+list(running_mean(dt, 20)))
        color = []
        max_mini_dt = max(abs(dt_grad))
        for dt in dt_grad:
            if abs(dt)<.001:
                c = cmapb
            elif dt<0:
                c = cmapg
            else: c = cmapr
            color.append(c.to_rgba(abs(dt/max_mini_dt)))
        return color

    return [cmapg.to_rgba(abs(dt/max_dt))
            if dt<0 else cmapr.to_rgba(dt/max_dt)
            for dt in dt]


def pedlascolor(df, ref=False):
    t = 'throttle'
    b = 'brake'
    if ref:
        t += '_r'
        b += '_r'

    color_pedals = []
    for thr, brk in zip(
            df[t].rolling(10).mean().values,
            df[b].rolling(10).mean().values):
        if abs(thr-brk)<10: c = cmapb
        elif thr>brk: c = cmapg
        else: c = cmapr
        color_pedals.append(c.to_rgba(max(thr,brk)/150))

    return color_pedals

def gloncolors(df, ref=False):
    g = 'g_lon'
    if ref: g += '_r'

    color_g_lon = []
    for g in df[g].rolling(10).mean().values:
        if g>0:
            color_g_lon.append(cmapg.to_rgba(g/max(df.g_lon)))
        else:
            color_g_lon.append(cmapr.to_rgba(abs(g)/max(df.g_lon.abs())))

    return color_g_lon


def addColorMaps(df, extra_maps=None):
    cmap = plt.get_cmap("jet")

    color_speed=cmap(1-df.speedkmh/300)
    color_g_lon=gloncolors(df)
    color_pedals=pedlascolor(df)

    color_speed_r=cmap(1-df.speedkmh_r/300, True)
    color_g_lon_r=gloncolors(df, True)
    color_pedals_r=pedlascolor(df, True)

    # convert colors to s.t. bokeh understands
    to_bokeh = lambda c: list(map(mplcolors.to_hex, c))
    for c in ['color_speed', 'color_pedals', 'color_g_lon',
              'color_speed_r', 'color_pedals_r', 'color_g_lon_r']:
        df = df.assign(**{c:pd.Series(to_bokeh(eval(c))).values})

    if extra_maps is not None:
        for c, v in extra_maps.items():
            df = df.assign(**{c:pd.Series(to_bokeh(v)).values})

    return df


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


def updateTableData(source, filter_source, track_select, car_select):
    data = scanFiles(glob.glob(os.environ['TELEMETRY_FOLDER']+'/*.ld'))

    source.data = data
    filter_source.data = copy.copy(data)

    getOptions = lambda key: \
        (['ALL'] + np.unique(source.data[key]).tolist()) \
            if key in source.data else ['ALL']

    track_select.options=getOptions('track')
    track_select.value = 'ALL'

    car_select.options=getOptions('car')
    car_select.value = 'ALL'