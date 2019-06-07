from bokeh.models.widgets import DataTable, TableColumn, Select
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.layouts import row



def create():
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

    return filters, data_table, source, filter_source, track_select, car_select