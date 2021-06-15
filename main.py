import os
import numpy as np

from bokeh.models.widgets import Button
from bokeh.models import Panel, Tabs, ColumnDataSource, Selection
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row

from upload import uploadButton
from ldparser import ldparser

import acctelemetry
import figures
import laptable


def callback():
    button.disabled = True
    s = []
    df = None
    idxs = filter_source.selected.indices

    for idx in idxs:
        # restrict to selected lap
        lap = int(filter_source.data['lap'][idx])
        name = filter_source.data['name'][idx]
        datetime = filter_source.data['datetime'][idx]
        time = filter_source.data['time'][idx]
        # track = filter_source.data['track'][idx]
        # carModel = filter_source.data['car'][idx]
        if len(name) > 3 and name[:3] == 'db:':
            import pymongo
            try:
                client = pymongo.MongoClient(os.environ['DB_HOST'], serverSelectionTimeoutMS=10)
                db = client.acc
            except pymongo.errors.ServerSelectionTimeoutError as err:
                print(err)
                continue

            name = name.split(':')
            ds = acctelemetry.DBDataStore(db, *name[1:], lap)

        else:
            f_ = os.path.join(os.environ['TELEMETRY_FOLDER'].strip("'"), filter_source.data['name'][idx])
            head_, chans = ldparser.read_ldfile(f_)

            laps = np.array(acctelemetry.laps(f_))
            ds = acctelemetry.LDDataStore(
                chans, laps,
                acc=head_.event!='AC_LIVE'
            )

        # create pandas DataFrame
        df_ = ds.get_data_frame(lap)
        # create a datasource and bind data
        s.append((datetime, lap, time,
                  ColumnDataSource.from_df(df_)))

        if df is None: df = df_
        else: df = df.append(df_)

    if df is None:
        button.disabled = False
        return

    # replace the figure
    figs[0].children[0] = figures.getFigure(s)
    figs[1].children[0] = figures.getRPMFigure(df)
    figs[2].children[0] = column([figures.getTyrePreassureFigure(df), figures.getTyreTairFigure(df)], sizing_mode='scale_width')
    figs[3].children[0] = column(figures.getWheelSpeedFigure(df), sizing_mode='scale_width')
    figs[4].children[0] = column(figures.getSuspFigure(df), sizing_mode='scale_width')
    figs[5].children[0] = column(figures.getBrakeTempFigure(df), sizing_mode='scale_width')
    figs[6].children[0] = figures.getOversteerFigure(df)
    figs[7].children[0] = column(figures.getTrackMapPanel(df), sizing_mode='scale_width')

    button.disabled = False


filters, data_table, source, filter_source, track_select, car_select = laptable.create()
acctelemetry.updateTableData(
    source, filter_source, track_select, car_select)

button = Button(label="Load", button_type="success")
button.on_click(callback)

tabs,figs = [],[]
tabs.append(Panel(
    child=column(filters, data_table, button, uploadButton(
        source, filter_source, track_select, car_select), sizing_mode='scale_width'),
    title="Laps"))

for ttl in ["LapData", "RPMs", "Tyre Preassure/TempAir", "WheelSpeed", "Suspension", "BrakeTemp", "Over/Understeer", "TrackMap"]:
    figs.append(row(figure(plot_height=500, plot_width=800), sizing_mode='scale_width', id=ttl.split(" ")[0].lower()))
    tabs.append(Panel(child=figs[-1], title=ttl, id="%spanel"%ttl.split(" ")[0].lower()))

tabs.append(Panel(child=figures.getLapDelta(), title="LapsDelta", id='lapsdeltapanel'))

tabs_ = Tabs(tabs=tabs, id='tabs')
curdoc().add_root(tabs_)

# filter_source.selected.indices = [0]
# callback()
# show(tabs_)
