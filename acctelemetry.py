import os, glob, copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm

import xml.etree.ElementTree as ET

from scipy import signal

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
    laps_limits = []
    if laps[0]!=0:
        laps_limits = [0]
    laps_limits.extend((np.array(laps)*freq).astype(int))
    laps_limits.extend([n])
    return list(zip(laps_limits[:-1], laps_limits[1:]))


def laps_times(laps):
    """calculate the laptime for each lap"""
    laps_times = []
    if len(laps) == 0: return laps_times
    if laps[0] != 0:  laps_times = [laps[0]]
    laps_times.extend(list(laps[1:]-laps[:-1]))
    return laps_times


# map from acti names to ACC names
ac_chan_map = {
    'ABS Active': 'abs',
    'Brake Pos': 'brake',
    'Brake Temp FL':'brake_temp_lf',
    'Brake Temp FR':'brake_temp_rf',
    'Brake Temp RL':'brake_temp_lr',
    'Brake Temp RR':'brake_temp_rr',
    'CG Accel Lateral':'g_lat',
    'CG Accel Longitudinal':'g_lon',
    'Engine RPM':'rpms',
    'Gear': 'gear',
    'Ground Speed':'speedkmh',
    'Steering Angle':'steerangle',
    'Suspension Travel FL':'sus_travel_lf',
    'Suspension Travel FR':'sus_travel_rf',
    'Suspension Travel RL':'sus_travel_lr',
    'Suspension Travel RR':'sus_travel_rr',
    'TC Active':'tc',
    'Throttle Pos':'throttle',
    'Wheel Angular Speed FL':'wheel_speed_lf',
    'Wheel Angular Speed FR':'wheel_speed_rf',
    'Wheel Angular Speed RL':'wheel_speed_lr',
    'Wheel Angular Speed RR':'wheel_speed_rr',
    'Tire Pressure FL':'tyre_press_lf',
    'Tire Pressure FR':'tyre_press_rf',
    'Tire Pressure RL':'tyre_press_lr',
    'Tire Pressure RR':'tyre_press_rr',
    'Tire Temp Core FL':'tyre_tair_lf',
    'Tire Temp Core FR':'tyre_tair_rf',
    'Tire Temp Core RL':'tyre_tair_lr',
    'Tire Temp Core RR':'tyre_tair_rr',
}

# map from pyacc shm names to ACC names
acc_shmem_map = {
    'packetId': 'packetId',
    'abs': 'abs',
    'brake': 'brake',
    'brakeTemp': ['brake_temp_lf',
                  'brake_temp_rf',
                  'brake_temp_lr',
                  'brake_temp_rr'],
    'accG': ['g_lat', 'accG', 'g_lon'],
    'rpms': 'rpms',
    'gear': 'gear',
    'roll': 'roll',
    'speedKmh': 'speedkmh',
    'steerAngle': 'steerangle',
    'suspensionTravel': ['sus_travel_lf',
                         'sus_travel_rf',
                         'sus_travel_lr',
                         'sus_travel_rr'],
    'tc': 'tc',
    'gas': 'throttle',
    'wheelSlip': ['wheel_slip_lf',
                  'wheel_slip_rf',
                  'wheel_slip_lr',
                  'wheel_slip_rr'],
    'wheelAngularSpeed': ['wheel_speed_lf',
                          'wheel_speed_rf',
                          'wheel_speed_lr',
                          'wheel_speed_rr'],
    'wheelsPressure': ['tyre_press_lf',
                       'tyre_press_rf',
                       'tyre_press_lr',
                       'tyre_press_rr'],
    'tyreContactPoint': ['tyre_contact_point_lf',
                         'tyre_contact_point_rf',
                         'tyre_contact_point_lr',
                         'tyre_contact_point_rr'],
    'tyreCoreTemperature': ['tyre_tair_lf',
                            'tyre_tair_rf',
                            'tyre_tair_lr',
                            'tyre_tair_rr'],
    'carDamage': ['damage_front',
                  'damage_rear',
                  'damage_left',
                  'damage_right',
                  'damage_centre']
}


class DataStore(object):
    @staticmethod
    def create_track(df, laps_times=None):
        # dx = (2*r*np.tan(alpha/2)) * np.cos(heading)
        # dy = (2*r*np.tan(alpha/2)) * np.sin(heading)
        dx = df.ds * np.cos(df.heading)
        dy = df.ds * np.sin(df.heading)

        # calculate correction to close the track
        # use best lap
        if laps_times is None:
            df_ = df
        else:
            fastest = np.argmin([999999 if x==0 else x for x in laps_times])
            df_ = df[(df.lap==fastest)]
        fac = 1.
        dist = None
        n = 0
        while n < 1000:
            dx = df_.ds * np.cos(df_.heading*fac)
            dy = df_.ds * np.sin(df_.heading*fac)
            end = (dx.cumsum()).values[-1], (dy.cumsum()).values[-1]
            # print(end, dist, fac)

            newdist = np.sqrt(end[0]**2+end[1]**2)
            if dist is not None and newdist>dist: break
            dist = newdist
            fac -= 0.0001
            n += 1

        if n == 1000:
            fac = 1.

        # recalculate with correction
        df.alpha = df.alpha*fac
        df.heading = df.alpha.cumsum()
        dx = df.ds * np.cos(df.heading*fac)
        dy = df.ds * np.sin(df.heading*fac)
        x = dx.cumsum()
        y = dy.cumsum()

        df = pd.concat([df, pd.DataFrame(
            {'x':x,'y':y,
             'dx':dx, 'dy':dy,
             })], axis=1)

        return df

    @staticmethod
    def calc_over_understeer(df):
        # calculate oversteer, based on math in ACC MoTec workspace
        wheelbase = 2.645
        neutral_steering = (wheelbase * df.alpha * 180/np.pi).rolling(10).mean()

        steering_corr= (df.steerangle/11)
        oversteer  = np.sign(df.g_lat) * (neutral_steering-steering_corr)
        understeer = oversteer.copy()

        indices = understeer > 0
        understeer[indices] = 0

        df = pd.concat([df, pd.DataFrame(
            {'steering_corr':steering_corr,
             'neutral_steering':neutral_steering,
             'oversteer':oversteer,
             'understeer':understeer})], axis=1)
        return df

    @staticmethod
    def add_cols(df, freq=None, n=None, laps_limits=None, lap=None):
        if 'speedkmh' not in df.columns:
            df = pd.concat([df, pd.DataFrame({'speedkmh': df.speed*3.6})], axis=1)
        if 'speed' not in df.columns:
            df = pd.concat([df, pd.DataFrame({'speed': df.speedkmh/3.6})], axis=1)

        # create list with the distance
        if 'ds' in df.columns:
            ds = df.ds
        else:
            ds = (df.speed / freq)
            df = pd.concat([df, pd.DataFrame({'ds': ds})], axis=1)

        # create list with total time
        if 'dt' in df.columns:
            t = df.dt.cumsum()
        else:
            t = np.arange(n)*(1/freq)

        # create list with the lap number, distance in lap, time in lap
        s = np.array(df.ds.cumsum())
        if laps_limits is None:
            l, sl, tl = [lap]*len(s), s, t
        else:
            l, sl, tl = [], [], []
            for n, (n1, n2) in enumerate(laps_limits):
                l.extend([n]*(n2-n1))
                sl.extend(list(s[n1:n2]-s[n1]))
                tl.extend(list(t[n1:n2]-t[n1]))

        # for calculate of x/y position on track from speed and g_lat
        gN = 9.81
        r = 1 / (gN * df.g_lat/df.speed.pow(2))
        alpha = ds / r
        heading = alpha.cumsum()

        # add the lists to the dataframe
        df = pd.concat([df, pd.DataFrame(
            {'lap':l,
             'g_sum': df.g_lon.abs()+df.g_lat.abs(),
             'heading':heading,
             'alpha':alpha,
             'dist':s,'dist_lap':sl,
             'time':t,'time_lap':tl})], axis=1)

        return df

    def get_data_frame(self, lap=None):
        pass


class DBDataStore(DataStore):
    def __init__(self, db, sid, start, end, lap, car_model):
        self.db = db
        self.sid = sid
        self.start = start
        self.end = end
        self.lap = lap
        self.car_model = car_model

    def get_data_frame(self, lap=None):
        from bson.objectid import ObjectId
        from pyacc import acc_types

        data = {}
        for v in acc_shmem_map.values():
            if isinstance(v, list):
                for _v in v:
                    data[_v] = []
            else:
                data[v] = []

        data['ds'] = []
        data['dt'] = []

        for p in self.db.physics.find({
            'sid': self.sid,
            '_id': {
                "$gte": ObjectId(self.start),
                "$lt": ObjectId(self.end)}}).sort('packedId'):

            # FIXME: frequency of packets seems to be 325 Hz ?
            dt = 0 if len(data['packetId']) == 0 else (p['packetId'] - data['packetId'][-1])*(1/325)
            ds = 0 if dt == 0 else (p['speedKmh']/3.6) * dt
            data['dt'].append(dt)
            data['ds'].append(ds)

            for k, v in acc_shmem_map.items():
                if isinstance(v, list):
                    if k == 'tyreContactPoint' and len(p[k]) == 3:
                        p[k] = np.array(p[k]).reshape((4, 3))
                    for i, _v in enumerate(v):
                        data[_v].append(p[k][i])
                else:
                    data[v].append(p[k])

        df = pd.DataFrame(data)
        # make scales comparable to those in motec files
        df.steerangle *= -1 * acc_types.maxSteeringAngle[getattr(acc_types.CAR_MODEL, self.car_model)]
        for i in ['throttle', 'brake']:
            df[i] *= 100
        for i in ['sus_travel_lf', 'sus_travel_rf', 'sus_travel_lr', 'sus_travel_rr']:
            df[i] *= 1000

        df = DataStore.add_cols(df, lap=lap)
        df = DataStore.calc_over_understeer(df)
        df = DataStore.create_track(df)

        if lap is not None:
            df = df[df.lap==lap]
        return df


class LDDataStore(DataStore):
    def __init__(self, channs, laps, acc=True):
        self.channs = channs
        self.acc = acc
        self.freq = 20
        self.n = self.freq * len(self.channs[0].data)//self.channs[0].freq
        self.columns = {}
        self._df = None
        self.laps_limits = laps_limits(laps, self.freq, self.n)
        self.laps_times = laps_times(laps)
        print('Scaling to %i Hz'%self.freq)

    def chan_name(self, x):
        if self.acc: return x.name.lower()
        return ac_chan_map[x.name] \
            if x.name in ac_chan_map \
            else x.name.lower()

    def __getitem__(self, item):
        if item not in self.columns:
            # print("Creating column %s"%(item))

            col = [n for n, x in enumerate(self.channs) if self.chan_name(x) == item]
            if len(col) != 1:
                raise Exception("Could not reliably get column", col)
            col = col[0]

            n = len(self.channs[col].data)
            x = np.linspace(0, n, self.n)
            data = np.interp(x, np.arange(0, n), self.channs[col].data)

            # convert some of the data from ld file to integer
            if (self.acc and col in [7, 11, 12]) or (not self.acc and col in [62]):
                data = data.astype(int)
            # downsample channels to the one with lowest frequency (this takes way tooo long)
            # if len(data) != self.n:
            #     data = signal.resample(data, self.n)

            self.columns[item] = data
        return self.columns[item]

    def get_data_frame(self, lap=None):
        for x in self.channs:
            _ = self[self.chan_name(x)]

        df = pd.DataFrame(self.columns)
        df = DataStore.add_cols(df, self.freq, self.n, self.laps_limits)
        df = DataStore.create_track(df)
        df = DataStore.calc_over_understeer(df)

        if lap is not None:
            df = df[df.lap==lap]
        return df


def lapdelta(reference, target):
    """
    # returns delta times against reference for target lap
    # - win of target vs reference -> green
    # - loss of target vs reference -> red
    :param df:          dataframe with the laps
    :param reference:   lap number of the reference lap
    :param target:      lap number of the lap to compare
    :return:            (list of delta times), df[lap==reference]
    """

    a, b = target, reference
    df_a = a[0].get_data_frame(a[1])
    df_b = b[0].get_data_frame(b[1])

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
    dt_, speed, speedkmh, throttle, brake, g_lon, xr, yr, oversteer = [],[],[],[],[],[],[],[],[]
    a_idx, b_idx = 0,0
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
        for i in ['speed', 'speedkmh', 'throttle', 'brake', 'g_lon', 'oversteer']:
            eval(i).append(df_b[i].values[b_idx])

    df_a = df_a.assign(dt=pd.Series(dt_).values)\
        .assign(xr=pd.Series(xr).values)\
        .assign(yr=pd.Series(yr).values)
    for i in ['speed', 'speedkmh', 'throttle', 'brake', 'g_lon', 'oversteer']:
        df_a = df_a.assign(**{'%s_r'%i:pd.Series(eval(i)).values})

    return df_a, df_b


def adddeltacolors(df, style=None):
    """
    Get colors for delta time
    :param dt:      the delta times
    :param style:   None: color maps to total dt, green -> faster, red -> slower
                    grad: color maps to the derivative of dt, red -> loosing, green -> gaining
    :return:        list of colors
    """
    dt = df.dt.rolling(20, min_periods=1).mean()
    if style == 'grad':
        dt = pd.Series(np.gradient(dt), index=df.index)
        m = dt.abs().max()
        b_ = dt[(dt.abs()<=.001)].map(lambda x:cmapb.to_rgba(x/m))
        r_ = dt[(dt.abs()>.001) & (dt>0)].abs().map(lambda x:cmapr.to_rgba(x/m))
        g_ = dt[(dt.abs()>.001) & (dt<=0)].abs().map(lambda x:cmapg.to_rgba(x/m))
        return df.assign(color_gainloss=pd.concat([b_,g_,r_]))

    m = dt.max()
    g_ = dt[(dt<0)].abs().map(lambda x:cmapg.to_rgba(x/m))
    r_ = dt[(dt>=0)].abs().map(lambda x:cmapr.to_rgba(x/m))
    return df.assign(color_absolut=pd.concat([g_,r_]))


def addpedalscolors(df, ref=False):
    t = 'throttle'
    b = 'brake'
    if ref:
        t += '_r'
        b += '_r'

    t_ = df[t].rolling(10, min_periods=1).mean()
    b_ = df[b].rolling(10, min_periods=1).mean()

    tc_ = t_.map(lambda x:cmapg.to_rgba(x/150))
    bc_ = b_.map(lambda x:cmapr.to_rgba(x/150))

    df = df.assign(**{'color_%s'%t:tc_})
    df = df.assign(**{'color_%s'%b:bc_})

    # tmp_ = pd.merge(t_, b_, left_index=True, right_index=True)
    # b_ = tmp_[((t_ - b_).abs() < 10)]
    # b0_ = b_[(t_<b_)][b].map(lambda x:cmapb.to_rgba(x/150))
    # b1_ = b_[(t_>=b_)][t].map(lambda x:cmapb.to_rgba(x/150))
    #
    # r_ = df[((t_ - b_).abs() >= 10) & (t_<b_)][b]
    # g_ = df[((t_ - b_).abs() >= 10) & (t_>=b_)][t]

    r_ = df[(t_<b_)]['color_%s'%b]
    g_ = df[(t_>=b_)]['color_%s'%t]

    n_ = 'pedals'
    if ref: n_ += '_r'
    return df.assign(**{'color_%s'%n_:pd.concat([g_,r_])})


def addgloncolors(df, ref=False):
    g = 'g_lon'
    if ref: g += '_r'

    g_lon = df[g].rolling(10, min_periods=1).mean()
    m0,m1 = g_lon.max(), g_lon.abs().max()
    g_ = g_lon[(g_lon>=0)].map(lambda x:cmapg.to_rgba(x/m0))
    r_ = g_lon[(g_lon<0)].abs().map(lambda x:cmapr.to_rgba(x/m1))

    return df.assign(**{'color_%s'%g:pd.concat([g_,r_])})


def addoversteercolors(df, ref=False):
    o = 'oversteer'
    if ref: o += '_r'

    oversteer = df[o].rolling(10, min_periods=1).mean()
    m0,m1 = oversteer.max(),abs(oversteer.min())
    r_ = oversteer[(oversteer>=0)].map(lambda x:cmapr.to_rgba(x/m0))
    b_ = oversteer[(oversteer<0)].abs().map(lambda x:cmapb.to_rgba(x/m1))

    return df.assign(**{'color_%s'%o:pd.concat([b_,r_])})


def addspeedcolors(df, ref=False):
    g = 'speedkmh'
    if ref: g += '_r'

    cmap = plt.get_cmap("jet")
    g_ = df[g].map(lambda x:cmap(1-x/300))
    return df.assign(**{'color_speed%s'%('_r' if ref else '') :g_})


def running_mean(x, N, min_periods=None):
    if min_periods is None: min_periods=N
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return np.append(x[:N-min_periods], (cumsum[N:] - cumsum[:-N]) / float(N))


def corners(df):
    g_lat = df.g_lat.rolling(50, 1).mean()
    # the g_lat gradient
    grad = running_mean(np.gradient(g_lat), 50, 1)
    # get zero crossings, indicating a max/min in g_lat -> apex of corner
    # ommit the first 50 samples
    zero_x = np.where(np.diff(np.sign(grad[50:])))[0]
    zero_x += 50
    # require a minimum g_lat
    # zero_x = np.extract(g_lat.abs().values[zero_x]>0.1, zero_x)
    keep = []
    for x in zero_x:
        # if max(g_lat.abs().values[x-50:x])<0.4: continue
        if g_lat.abs().values[x-50:x].mean()<0.4: continue
        keep.append(x)
    zero_x = keep

    # require that successive corners have eather a different direction
    new_zero_x = []
    for x in zero_x:
        corners = np.extract( abs(zero_x-x) < 100, zero_x)
        corners = np.extract( corners!=x, corners)
        # no close corners
        if len(corners)==0:
            new_zero_x.append(x)
            continue

        # check the direction of the corners
        corners_dir = g_lat.values[corners]
        corners_same_dir = np.extract(
            np.sign(corners_dir)==np.sign(g_lat.values[x]), corners)

        # the corners are different direction than the current one
        if len(corners_same_dir) != len(corners):
            new_zero_x.append(x)
            continue

        # the corners are same direction, use the strongest one
        if abs(g_lat.values[x])>max(abs(corners_dir)):
            new_zero_x.append(x)
            continue

    return np.array(new_zero_x)-50


def scanFiles(files):
    data = []
    for f in files:
        if not os.path.isfile(os.path.splitext(f)[0]+".ldx"): continue
        head = ldparser.ldHead.fromfile(open(f,'rb'))
        laps_ = laps_times(np.array(laps(f)))
        for i, lap in enumerate(laps_):
            if lap==0: continue
            data.append((os.path.basename(f),
                         head.datetime,
                         head.venue, head.event, i,
                         "%i:%02i.%03i"%(lap//60, lap%60, (lap*1e3)%1000),
                         head.driver,
                         ))
    return data


def scanDB(db):
    connections = db.static.aggregate(
        [{'$group':
             {'_id': {
                 'sid': '$sid',
                 'track': '$track',
                 'carModel': '$carModel',
                 'playerNick': '$playerNick',
                 'playerName': '$playerName',
                 'playerSurname': '$playerSurname',
             },
                 "num_statics" : {"$sum": 1},
                 "min_id": {"$min": '$_id'},
                 "max_id": {"$max": '$_id'},
             }},
             {'$sort': {
                "min_id": -1,
                "_id.playerSurname": 1,
                "_id.playerName": 1,
             }},
            {'$match': {'num_statics': {'$gt': 300}}},  # require a minimum of 5mins recorded time
         ])

    data = []
    for con in connections:
        laps = db.graphics.aggregate(
            [{'$match':
                  {'sid': con['_id']['sid'],
                   'completedLaps': {"$gt": 0},
                   '_id': {"$gte": con['min_id'],
                           "$lt": con['max_id']}
                   }},
             {'$group':
                 {'_id': {
                     'session': '$session',
                     'lap': '$completedLaps',
                 },
                     'iLastTime': {'$max': '$iLastTime'},
                     'min_id': {'$min': '$_id'},
                     'max_id': {'$max': '$_id'},
                 }},
             {'$sort': {'_id': -1}},
             ])

        # check if there's data
        if not laps.alive:
            continue

        _time, _lap = None, None
        for l in laps:
            # first lap or 'jump' in lap count - don't store data of this lap, save
            # laptime of preceding lap
            if _lap is None or l['_id']['lap'] != _lap - 1:
                _time = None

            if _time is not None:
                data.append(('db:%s:%s:%s' % (
                    con['_id']['sid'], l['min_id'], l['max_id']),
                    con['min_id'].generation_time.replace(tzinfo=None),
                    con['_id']['track'],
                    con['_id']['carModel'], l['_id']['lap'],
                    "%i:%02i.%03i"%(_time//60, _time%60, (_time*1e3) % 1000),
                     con['_id']['playerNick']  if len(con['_id']['playerNick']) > 0
                        else "%s %s" % (con['_id']['playerName'], con['_id']['playerSurname']),
                    ))
            _time = l['iLastTime']/1000
            _lap = l['_id']['lap']

    return data


def updateTableData(source, filter_source, track_select, car_select):
    data = scanFiles(glob.glob(os.path.join(os.environ['TELEMETRY_FOLDER'].strip("'"), '*.ld')))

    if 'DB_HOST' in os.environ:
        import pymongo
        try:
            client = pymongo.MongoClient(os.environ['DB_HOST'], serverSelectionTimeoutMS=10)
            client.server_info()
            db = client.acc
            data.extend(scanDB(db))
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print('DB not available', err)
            pass

    data = np.array(sorted(data, key=lambda x: (x[1], x[6], x[4]), reverse=True))
    data = dict(
        name=data[:, 0],
        datetime=[d.strftime("%Y-%m-%d %H:%M:%S") for d in data[:, 1]],
        track=data[:, 2],
        car=data[:, 3],
        lap=data[:, 4],
        time=data[:, 5],
        driver=data[:, 6]
    )
    source.data = data
    filter_source.data = copy.copy(data)

    getOptions = lambda key: \
        (['ALL'] + list(map(str, np.unique(source.data[key])))) \
            if key in source.data else ['ALL']

    track_select.options=getOptions('track')
    track_select.value = 'ALL'

    car_select.options=getOptions('car')
    car_select.value = 'ALL'
