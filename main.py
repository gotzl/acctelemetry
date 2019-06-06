import os, glob
import numpy as np

from bokeh.models.widgets import DataTable, TableColumn, Button, Select
from bokeh.models import Panel, Tabs, ColumnDataSource, CustomJS
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row

from upload import uploadButton
from ldparser import ldparser
import acctelemetry
import figures


def callback():
    button.disabled = True
    s = []
    df = None
    idxs = filter_source.selected.indices
    # idxs = source.selected['1d']['indices']

    for idx in idxs:
        f_ = os.environ['TELEMETRY_FOLDER']+'/%s'%filter_source.data['name'][idx]
        head_, chans = ldparser.read_ldfile(f_)

        laps = acctelemetry.laps(f_)
        laps_limits = acctelemetry.laps_limits(laps, chans[4].freq, len(chans[4].data))


        # create pandas DataFrame
        df_ = acctelemetry.createDataFrame(chans, laps_limits)
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



columns = [
    TableColumn(field="name", title="File name"),
    TableColumn(field="datetime", title="Datetime"),
    TableColumn(field="track", title="Track"),
    TableColumn(field="car", title="Car"),
    TableColumn(field="lap", title="Lap"),
    TableColumn(field="time", title="Lap time"),
]

source = ColumnDataSource()
filter_source = ColumnDataSource()
data_table = DataTable(source=filter_source, columns=columns, width=800)

### https://gist.github.com/dennisobrien/450d7da20daaba6d39d0
# callback code to be used by all the filter widgets
combined_callback_code = """
var data = filter_source.data;
var original_data = source.data;
var track = track_select_obj.value;
var car = car_select_obj.value;
for (var key in original_data) {
    data[key] = [];
    for (var i = 0; i < original_data['track'].length; ++i) {
        if ((track === "ALL" || original_data['track'][i] === track) &&
                (car === "ALL" || original_data['car'][i] === car)) {
            data[key].push(original_data[key][i]);
        }
    }
}

filter_source.change.emit();
target_obj.change.emit();
"""

# define the filter widgets, without callbacks for now
track_select = Select(title="Track:", value='ALL', options=['ALL'])
car_select = Select(title="Car:", value='ALL', options=['ALL'])

# now define the callback objects now that the filter widgets exist
generic_callback = CustomJS(
    args=dict(source=source,
              filter_source=filter_source,
              track_select_obj=track_select,
              car_select_obj=car_select,
              target_obj=data_table),
    code=combined_callback_code
)

# finally, connect the callbacks to the filter widgets
track_select.js_on_change('value', generic_callback)
car_select.js_on_change('value', generic_callback)
filters = row(track_select, car_select)
######


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
    tabs.append(Panel(child=figs[-1], title=ttl))

tabs_ = Tabs(tabs=tabs)

curdoc().add_root(tabs_)

# source.selected = {'0d': {'glyph': None, 'indices': []},
#                    '1d': {'indices': [0]}, '2d': {}}
# callback()
# show(tabs_)
