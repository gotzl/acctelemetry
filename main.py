import os, glob

from bokeh.models.widgets import DataTable, TableColumn, Button
from bokeh.models import Panel, Tabs, ColumnDataSource
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row

from upload import uploadButton
from ldparser import ldparser
import acctelemetry
import figures

base_ = os.environ['TELEMETRY_FOLDER']
files = glob.glob(base_+'/*.ld')
source = acctelemetry.createSource(files)


def callback():
    s = []
    df = None
    for idx in source.selected['1d']['indices']:
        f_ = base_+'/%s'%source.data['name'][idx]
        head_, chans = ldparser.read_ldfile(f_)

        laps = acctelemetry.laps(f_)
        laps_limits = acctelemetry.laps_limits(laps, chans[4].freq, len(chans[4].data))


        # create pandas DataFrame
        df_ = acctelemetry.createDataFrame(chans, laps_limits)
        # restrict to selected lap
        lap = int(source.data['lap'][idx])
        df_ = df_[df_.lap==lap]

        # create a datasource and bind data
        s.append( (head_.datetime, lap,
                   laps[lap] if lap==0 else laps[lap]-laps[lap-1],
                   ColumnDataSource(df_)))

        if df is None: df = df_
        else: df.append(df_)

    if df is None: return

    # replace the figure
    figs[0].children[0] = figures.getFigure(s)
    figs[1].children[0] = figures.getRPMFigure(df)
    figs[2].children[0] = figures.getWheelSpeedFigure(df)
    figs[3].children[0] = figures.getOversteerFigure(df)
    figs[4].children[0] = figures.getSuspFigure(df)


columns = [
    TableColumn(field="name", title="File name"),
    TableColumn(field="datetime", title="Datetime"),
    TableColumn(field="location", title="Location"),
    TableColumn(field="car", title="Car"),
    TableColumn(field="lap", title="Lap"),
    TableColumn(field="time", title="Lap time"),
]


data_table = DataTable(source=source, columns=columns, width=800)
button = Button(label="Load", button_type="success")
button.on_click(callback)

tabs,figs = [],[]
tabs.append(Panel(child=column(data_table, button, uploadButton(base_, data_table)), title="Laps"))
for ttl in ["LapData", "RPMs", "Wheelspeed", "Over/Understeer", "Susp Trvl"]:
    figs.append(row(column(figure(plot_height=500, plot_width=800))))
    tabs.append(Panel(child=figs[-1], title=ttl))

tabs_ = Tabs(tabs=tabs)

curdoc().add_root(tabs_)

# source.selected = {'0d': {'glyph': None, 'indices': []},
#                    '1d': {'indices': [0]}, '2d': {}}
# callback()
# show(tabs_)
