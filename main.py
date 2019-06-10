import os
import numpy as np

from bokeh.models.widgets import Button, TextInput, TextAreaInput
from bokeh.models import Panel, Tabs, ColumnDataSource, CustomJS
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
        f_ = os.environ['TELEMETRY_FOLDER']+'/%s'%filter_source.data['name'][idx]
        head_, chans = ldparser.read_ldfile(f_)

        laps = acctelemetry.laps(f_)
        laps_limits = acctelemetry.laps_limits(laps, chans[4].freq, len(chans[4].data))

        laps = np.array(laps)
        laps_times = [laps[0]]
        laps_times.extend(list(laps[1:]-laps[:-1]))

        # create pandas DataFrame
        df_ = acctelemetry.createDataFrame(
            filter_source.data['name'][idx], chans, laps_times, laps_limits)
        # restrict to selected lap
        lap = int(filter_source.data['lap'][idx])
        df_ = df_[df_.lap==lap]

        # create a datasource and bind data
        s.append( (head_.datetime, lap,
                   laps[lap] if lap==0 else laps[lap]-laps[lap-1],
                   ColumnDataSource(df_)))

        if df is None: df = df_
        else: df = df.append(df_)

    if df is None:
        button.disabled = False
        return

    # replace the figure
    figs[0].children[0] = figures.getFigure(s)
    figs[1].children[0] = figures.getRPMFigure(df)
    figs[2].children[0] = figures.getWheelSpeedFigure(df)
    figs[3].children[0] = figures.getOversteerFigure(df)
    figs[4].children[0] = figures.getSuspFigure(df)
    button.disabled = False


filters, data_table, source, filter_source, track_select, car_select = laptable.create()
acctelemetry.updateTableData(
    source, filter_source, track_select, car_select)

button = Button(label="Load", button_type="success")
button.on_click(callback)

tabs,figs = [],[]
tabs.append(Panel(
    child=column(filters, data_table, button, uploadButton(
        source, filter_source, track_select, car_select)),
    title="Laps"))

for ttl in ["LapData", "RPMs", "Wheelspeed", "Over/Understeer", "Susp Trvl"]:
    figs.append(row(figure(plot_height=500, plot_width=800)))
    tabs.append(Panel(child=figs[-1], title=ttl, id=ttl.split(" ")[0].lower()))

tabs.append(Panel(child=figures.getLapDelta(), title="LapsDelta", id='lapsdeltapanel'))

tabs_ = Tabs(tabs=tabs, id='tabs')
curdoc().add_root(tabs_)

# source.selected = {'0d': {'glyph': None, 'indices': []},
#                    '1d': {'indices': [0]}, '2d': {}}
# callback()
# show(tabs_)
