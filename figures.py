import os, itertools
import numpy as np
from bokeh.io import curdoc
import matplotlib.colors as mplcolors

from bokeh.palettes import Spectral4, Dark2_5 as palette
from bokeh.plotting import figure
from bokeh.models import CrosshairTool, HoverTool, CustomJS, \
    LinearAxis, ColumnDataSource, Range1d, TextInput, Circle, Select, Line, Button, Selection, Slider, TapTool, \
    LabelSet, Text
from bokeh.layouts import gridplot, column, row

import acctelemetry, laptable
from ldparser import ldparser


def createHoverTool(tools, mode='mouse'):
    _tools = dict(
        dist=("Dist", "@dist_lap{%0.1f} m"),
        time=("Time", "@time_lap{%0.1f} s"),
        dt=("Dt", "@dt{%0.3f} s"),
        speed=("Speed", "@speed m/s"),
        speedkmh=("Speed", "@speedkmh km/h"),
        rpms=("RPMs","@rpms"),
        g_lon=("G lon","@g_lon"),
        g_lat=("G lat","@g_lat"),
    )

    _tool = lambda x: _tools[x] if x in _tools else (x.title(),'@%s'%x)

    return HoverTool(
        tooltips=[_tool(x) for x in tools],
        formatters={'dist' : 'printf',
                    'time' : 'printf',
                    'dt' : 'printf',
                    'dist_lap' : 'printf',
                    'time_lap' : 'printf',
                    },
        mode=mode)


def getFigure(sources, x='dist_lap', width=800):
    TOOLS = "crosshair,pan,reset,save,wheel_zoom"
    # TOOLS = "pan,reset,save,wheel_zoom"

    # define height of each figure
    heights= [
        400,150,150,100,150,150
    ]

    # define data for each figure
    ys = [
        ['speedkmh'],
        ['steerangle'],
        ['rpms'],
        ['gear'],
        ['throttle','brake'],
        ['g_lon','g_lat','g_sum']
    ]

    # define the tooltips
    tools = [
        [
            ("Time", "@time_lap{%0.1f} s"),
            ("Dist", "@dist_lap{%0.1f} m"),
            ("Speed", "@speedkmh km/h")],
        [("Steerangle","@steerangle")],
        [("RPMs","@rpms")],
        [("Gear","@gear")],
        [("Throttle","@throttle"), ("Brake","@brake")],
        [("G lon","@g_lon"), ("G lat","@g_lat"), ("G sum","@g_sum")],
    ]

    # some JS needed to link crosshairs
    # TODO: replace with
    # https://stackoverflow.com/questions/37965669/how-do-i-link-the-crosshairtool-in-bokeh-over-several-plots
    js_move = '''
        if(cb_obj.x >= fig.x_range.start && cb_obj.x <= fig.x_range.end &&
           cb_obj.y >= fig.y_range.start && cb_obj.y <= fig.y_range.end)
        {
            c.spans.height.computed_location = cb_obj.sx
        }
        else
        {
            c.spans.height.computed_location = null
        }
    '''
    js_leave = 'c.spans.height.computed_location = null'


    # some JS needed to link tooltips
    # code = "source.set('selected', cb_data['index']);"
    # callback = CustomJS(args={'source': source}, code=code)

    colors = itertools.cycle(palette)
    muted_colors = itertools.cycle(Spectral4)
    p,c,h = [],[],[]
    for height, y, tool in zip(heights, ys, tools):
        p_ = figure(plot_height=height, plot_width=width, tools=TOOLS,
                    title=None, x_axis_label='Dist [m]', y_axis_label='|'.join(y))

        # creat crosshair
        c_ = CrosshairTool()

        # link the crosshairs together
        for cc_ in c:
            args = {'c': cc_, 'fig': p_}
            p_.js_on_event('mousemove', CustomJS(args=args, code=js_move))
            p_.js_on_event('mouseleave', CustomJS(args=args, code=js_leave))
        for pp_ in p:
            args = {'c': c_, 'fig': pp_}
            pp_.js_on_event('mousemove', CustomJS(args=args, code=js_move))
            pp_.js_on_event('mouseleave', CustomJS(args=args, code=js_leave))

        # toolbar ontop for the smaller plots at the bottom
        if len(p)>0:
            p_.x_range = p[0].x_range
            p_.toolbar_location="above"

        # creat tooltip
        h_ = HoverTool(
            tooltips=tool if len(h)==0 else tools[0]+tool,
            formatters={
                'dist': 'printf',
                'dist_lap': 'printf',
                'time_lap': 'printf',
            },
            mode='mouse' if len(y)>1 or len(sources)>1 else 'vline')

        p_.add_tools(c_)
        p_.add_tools(h_)

        # create the actual plot;
        # if multiple sources, use same color for all plots of one source
        if (len(sources)>1):
            colors = itertools.cycle(palette)
            muted_colors = itertools.cycle(Spectral4)

        for datetime, lap, lap_t, source in sources:
            lap_t = "%i:%02i.%03i"%(lap_t//60,lap_t%60,(lap_t*1e3)%1000)
            for yi in y:
                p_.line(x=x, y=yi, source=source,
                        legend='%s | lap %s | %s'%(datetime, lap, lap_t) if len(p)==0 else '',
                        muted_color=next(muted_colors), muted_alpha=0.2,
                        line_width=3, line_alpha=0.6,
                        line_color=next(colors))

        p_.legend.location = "top_right"
        p_.legend.click_policy="mute"

        p.append(p_)
        c.append(c_)
        h.append(h_)

    return gridplot(p, ncols=1)


def getRPMFigure(df):
    WIDTH = 800
    TOOLS = "crosshair,pan,reset,save,wheel_zoom"

    # create a new plot with a title and axis labels
    p1 = figure(plot_height=400, plot_width=WIDTH, tools=TOOLS,
                x_axis_label='Velocity [km/h]', y_axis_label='RPMs [1/s]')

    # add a line renderer with legend and line thickness
    colors = itertools.cycle(palette)
    df_ = df.groupby('gear')
    for grp, df_ in df_:
        col = next(colors)
        p1.circle(df_.speedkmh, df_.rpms, legend='gear %i'%grp,
                  muted_color=col, muted_alpha=0.1,
                  size=3, color=col, line_color=None)

    p1.legend.location = "top_right"
    p1.legend.click_policy="mute"


    # create a new plot with a title and axis labels
    p2 = figure(plot_height=400, plot_width=WIDTH, tools=TOOLS,
                x_range = p1.x_range, x_axis_label='Velocity [km/h]', y_axis_label='G longi [m/s^2]')

    # add a line renderer with legend and line thickness
    colors = itertools.cycle(palette)
    for i in range(1,7):
        sel = df[ (df.gear==i) & (df.throttle>80)]
        col = next(colors)
        p2.circle(sel.speedkmh, sel.g_lon, legend='gear %i'%i,
                  muted_color=col, muted_alpha=0.1,
                  size=3, color=col, line_color=None)

    p2.legend.location = "top_right"
    p2.legend.click_policy="mute"

    return column(p1,p2)


def getSimpleFigure(df, vars, tools, extra_y=None, extra_y_vars=None, x_range=None):
    WIDTH = 800
    TOOLS = "crosshair,pan,reset,save,wheel_zoom"

    # create a new plot with a title and axis labels
    p = figure(plot_height=400, plot_width=WIDTH, tools=TOOLS,
               x_range=x_range, x_axis_label='Dist [m]')

    y_range_name = lambda x: None
    if extra_y:
        # Setting the second y axis range name and range
        p.extra_y_ranges = extra_y

        def y_range_name(var):
            # create mapping of variable to axis
            if extra_y_vars is None:
                vars = list(zip(extra_y.keys(), extra_y.keys()))
            else:
                vars = [(var, ax) for ax, sublist in extra_y_vars.items() for var in sublist]

            # check if a variable in the vars list matches the argument
            k = [k for k in vars if k[0] in var]
            if len(k)==1:
                return k[0][1]
            return 'default'

        for name in extra_y:
            # Adding the second axis to the plot.
            p.add_layout(LinearAxis(
                y_range_name=name,
                axis_label=name), 'right')


    colors = itertools.cycle(palette)
    ds = ColumnDataSource(df)
    for i in vars:
        p.line(x='dist_lap', y=i, source=ds,
               legend='m = {}'.format(i),
               line_width=2, line_alpha=0.6,
               line_color=next(colors),
               y_range_name=y_range_name(i)
               )

    p.toolbar_location="above"
    p.add_tools(createHoverTool(tools))

    return p


def getSuspFigure(df):
    vars = ['speedkmh', 'sus_travel_lf', 'sus_travel_lr',
            'sus_travel_rf', 'sus_travel_rr']
    tools = ['time', 'dist']+vars
    return getSimpleFigure(df, vars, tools,
                           {"sus_travel": Range1d(start=-10, end=120)})

def getBrakeTempFigure(df):
    vars = ['brake_temp_lf','brake_temp_lr',
            'brake_temp_rf', 'brake_temp_rr',
            'throttle','brake']
    tools = ['time', 'dist', 'speed']+vars
    return getSimpleFigure(df, vars+['tc', 'abs'], tools,
                           {"pedals": Range1d(start=-20, end=400), "tcabs": Range1d(start=-1, end=20)},
                           {"pedals":['throttle','brake'], "tcabs":['tc', 'abs']})

def getWheelSpeedFigure(df):
    vars = ['wheel_speed_lf','wheel_speed_lr',
            'wheel_speed_rf', 'wheel_speed_rr',
            'throttle','brake']
    tools = ['time','dist','speed']+vars
    return getSimpleFigure(df, vars+['tc', 'abs'], tools,
                           {"pedals": Range1d(start=-20, end=400), "tcabs": Range1d(start=-1, end=20)},
                           {"pedals":['throttle','brake'], "tcabs":['tc', 'abs']})

def getTyreTairFigure(df):
    vars = ['tyre_tair_lf','tyre_tair_lr',
            'tyre_tair_rf', 'tyre_tair_rr',
            'throttle','brake']
    tools = ['time', 'dist', 'speed']+vars
    return getSimpleFigure(df, vars+['tc', 'abs'], tools,
                           {"pedals": Range1d(start=-20, end=400), "tcabs": Range1d(start=-1, end=20)},
                           {"pedals":['throttle','brake'], "tcabs":['tc', 'abs']})

def getTyrePreassureFigure(df):
    vars = ['speedkmh', 'tyre_press_lf','tyre_press_lr',
            'tyre_press_rf', 'tyre_press_rr']
    tools = ['time', 'dist']+vars
    return getSimpleFigure(df, vars, tools,
                           {"tyre_press": Range1d(start=26, end=32)})

def getOversteerFigure(df):
    vars = ['speedkmh','oversteer','understeer']#,'steering_corr','neutral_steering']
    tools = ['time','dist']+vars
    p0 =  getSimpleFigure(df, vars+['tc', 'throttle','brake'], tools,
                           {"pedals": Range1d(start=-10, end=500), "tc": Range1d(start=-1, end=50), "oversteer": Range1d(start=-15, end=25)},
                           {"pedals":['throttle','brake'], "tc":['tc','g_lat'], "oversteer":['steering_corr','neutral_steering','oversteer','understeer']})

    vars = ['g_lat', 'g_lon', 'g_sum','steering_corr','neutral_steering', 'oversteer', 'understeer']
    tools = ['time','dist']+vars
    p1 =  getSimpleFigure(df, vars, tools,
                          {"oversteer": Range1d(start=-15, end=35)},
                          {"oversteer":['steering_corr','neutral_steering',
                                        'oversteer','understeer']},
                          x_range = p0.x_range)

    return gridplot([p0,p1], ncols=1)


def getLapDelta():
    filters, data_table, source, filter_source, track_select, car_select = laptable.create()
    acctelemetry.updateTableData(
        source, filter_source, track_select, car_select)

    def callback_(attrname, old, new):
        callback(mode_select.value)

    def callback(mode):
        idxs = filter_source.selected.indices
        if (len(idxs)<2):
            layout.children[-1] = tmp
            return

        df, track, reference, target = None, None, None, None
        for idx in idxs:
            name = filter_source.data['name'][idx]
            f_ = os.path.join(os.environ['TELEMETRY_FOLDER'].strip("'"), name)
            head_, chans = ldparser.read_ldfile(f_)

            # if track is not None and head_.descr1!=track: continue

            laps = np.array(acctelemetry.laps(f_))
            laps_limits = acctelemetry.laps_limits(laps, chans[4].freq, len(chans[4].data))
            laps_times = acctelemetry.laps_times(laps)

            # create DataStore that is used later to get pandas DataFrame
            ds = acctelemetry.DataStore(
                chans, laps_times, laps_limits, acc=head_.event!='AC_LIVE')
            # restrict to selected lap
            lap = int(filter_source.data['lap'][idx])

            info = [ds, lap, head_.vehicle, laps_times[lap]]
            if track is None:
                # df = df_
                reference = info
                track = head_.venue
            else:
                # df = df.append(df_)
                target = info

        if reference is None or target is None:
            layout.children[-1] = tmp
            return

        # text_input.value = "%s: reference: %s (%i) | target: %s (%i)"%(track, reference, idxs[0], target, idxs[-1])
        text_input.value = "%s | %s: reference: %s / %.3f (%i) | target: %s / %.3f (%i)"%\
                           (track, mode, reference[2], reference[3], idxs[0],
                            target[2], target[3], idxs[-1])

        layout.children[-1] = getTrackMap(target[:2], reference[:2], mode)

    def mode_change(attrname, old, new):
        callback(new)

    text_input = TextInput(value="nothing selected")
    text_input.disabled = True

    mode_select = Select(title="Mode:", value='absolut',
                         options=['absolut',
                                  'gainloss',
                                  'oversteer',
                                  'speed',
                                  'pedals',
                                  'throttle',
                                  'brake',
                                  'g_lon'])
    mode_select.on_change('value', mode_change)

    filter_source.selected.on_change('indices', callback_)
    tmp = figure(plot_height=500, plot_width=800)
    layout = column(filters, data_table, mode_select, text_input, tmp, id='lapsdelta')
    return layout


color_mode_map = {'absolut': acctelemetry.adddeltacolors,
                  'gainloss': lambda x,_: acctelemetry.adddeltacolors(x, 'grad'),
                  'g_lon': acctelemetry.addgloncolors,
                  'oversteer': acctelemetry.addoversteercolors,
                  'speed':acctelemetry.addspeedcolors,
                  'pedals':acctelemetry.addpedalscolors,
                  'throttle':acctelemetry.addpedalscolors,
                  'brake':acctelemetry.addpedalscolors,
                  }


def getLapFigure(p1, df_, ds, mode, ref=False, hasref=False):
    # add required colors to dataframe and create datasource
    df_ = color_mode_map[mode](df_, ref)

    to_bokeh = lambda c: list(map(mplcolors.to_hex, c))

    x = 'xr' if ref else 'x'
    y = 'yr' if ref else 'y'
    color = 'color_%s'%((mode+'_r') if ref else mode)
    ds.data[color] = to_bokeh(df_[color])

    # shift the reference points to the outside
    if ref:
        ds.data[x] += 30*np.cos(df_.heading+np.pi/2)
        ds.data[y] += 30*np.sin(df_.heading+np.pi/2)

    # plot the track map, overwrite the (non)selection glyph to keep our color from ds
    # the hover effect is configured below
    r2 = p1.scatter(x=x, y=y, source=ds, color=color)
    r2.nonselection_glyph = r2.selection_glyph

    if ref: return p1

    # add some lap descriptions
    corners = acctelemetry.corners(df_)
    corners_ds = ColumnDataSource(dict(
        x=df_.x.values[corners],
        y=df_.y.values[corners],
        text=['T%i'%i for i in range(1, len(corners)+1)],
    ))
    labels = LabelSet(x='x', y='y', text='text', level='glyph',
                      x_offset=5, y_offset=5,
                      source=corners_ds, render_mode='canvas')
    p1.add_layout(labels)

    # create a invisible renderer for the track map
    # this is used to trigger the hover, thus the size is large
    c1 = p1.circle(x='x', y='y', source=ds, size=10, fill_alpha=0.0, alpha=0.0)
    c1.selection_glyph = Circle(fill_color='red', fill_alpha=.7, line_color=None)
    c1.nonselection_glyph = Circle(fill_alpha=0, line_color=None)

    # create a renderer to show a dot for the reference
    if hasref:
        cr = p1.circle(x='xr', y='yr', source=ds,
                       size = 8 if mode in ['absolut', 'gainloss'] else 10,
                       fill_alpha=0.0, alpha=0.0)
        cr.selection_glyph = Circle(fill_color='blue', fill_alpha=.7, line_color=None)
        cr.nonselection_glyph = Circle(fill_alpha=0, line_color=None)

    return c1


def getLapSlider(ds, p0, r0, hover0, view):
    # Enable selection update with slider
    slider = Slider(start=0, end=len(ds.data['dist_lap']),value=0, step=50)

    # React on changes of the selection in the datasource. Display tooltips at the position of the selected point.
    code = """
    let ind = slider.value;
    let x = source.data.dist_lap[ind];
    let y = source.data.speedkmh[ind];
    let fig_view;
    if (view == "trackmap") 
        fig_view = Bokeh.index["tabs"]
            ._child_views[view]
            .child_views[0]
            .child_views[1]
            ._child_views[figure.id];
    if (view == "lapsdelta") {
        var lapsdelta_view = Bokeh.index["tabs"]._child_views[view].child_views;
        fig_view = lapsdelta_view[lapsdelta_view.length-1]._child_views[figure.id];
    }
    let hover_view = fig_view.tool_views[hovertool.id];
    let renderer_view = fig_view.renderer_views[renderer.id];
    let xs = renderer_view.xscale.compute(x);
    let ys = renderer_view.yscale.compute(y);
    hover_view._inspect(xs, ys);
    source.selected.indices = [ind]; // this triggers c0/c1/cr selected glyph
    """
    callback = CustomJS(args=dict(hovertool=hover0,
                                  source=ds,
                                  figure=p0,
                                  view=view,
                                  slider=slider,
                                  renderer=r0), code=code)
    slider.js_on_change('value', callback)
    return slider


def getLapControls(ds, slider):

    #### create a player that iterates over the data
    def increment(stepsize, direction=1):
        i = slider.value + direction*stepsize
        if i<0: i = 0
        if i>len(ds.data['dist']): i = len(ds.data['dist'])-1
        # update of the selected item in the ds by modifying the slider value
        slider.value = i
        slider.trigger('value', slider.value, i)


    def cbcrl(stop=False):
        global cb

        if cb is None and not stop:
            cb = curdoc().add_periodic_callback(lambda: increment(5), 5*50)
            # reset hovertool
            # hover0.renderers = []
            # hover.renderers = []
            play.label = "Pause"
        else:
            try:  curdoc().remove_periodic_callback(cb)
            except: pass
            cb = None
            play.label = "Play"

    global cb
    if not 'cb' in globals():
        cb = None
    if cb is not None:
        try:  curdoc().remove_periodic_callback(cb)
        except: cb = None

    play = Button(label="Play")
    play.on_click(cbcrl)

    btns = []
    def bb(): increment(50, -1)
    def b(): increment(5, -1)
    def f(): increment(5, 1)
    def ff(): increment(50, 1)
    for l,s in [('<<',bb),('<',b),('>',f),('>>',ff)]:
        b = Button(label=l, width=50)
        b.on_click(s)
        btns.append(b)

    return row(play, *btns)


def getTrackMap(target, reference=None, mode='speed', view='lapsdelta'):
    if reference is None:
        df_, df_r = target, None
    else:
        df_, df_r = acctelemetry.lapdelta(reference, target)

    ds = ColumnDataSource(df_)
    p0 = figure(plot_height=400, plot_width=800,
                tools="crosshair,pan,reset,save,wheel_zoom")

    colors = itertools.cycle(palette)
    col0, col1 = next(colors), next(colors)

    # create the velo vs dist plot
    r0 = p0.line(x='dist_lap', y='speedkmh', source=ds, color=col0, line_width=2)

    # overwrite the (non)selection glyphs with the base line style
    # the style for the hover will be set below
    nonselected_ = Line(line_alpha=1, line_color=col0, line_width=2)
    r0.selection_glyph = nonselected_
    r0.nonselection_glyph = nonselected_

    if reference is not None:
        # create the dt vs dist plot with extra y axis, set the (non)selection glyphs
        lim = max(df_.dt.abs())
        lim += lim*.2
        p0.extra_y_ranges = {"dt": Range1d(start=-lim, end=lim)}
        p0.add_layout(LinearAxis(
            y_range_name='dt',
            axis_label='dt [s]'), 'right')
        r1 = p0.line(x='dist_lap', y='dt', source=ds, y_range_name='dt', color=col1, line_width=2)
        r1.selection_glyph = Line(line_alpha=1, line_color='red', line_width=5)
        r1.nonselection_glyph = Line(line_alpha=1, line_color=col1, line_width=2)

        # create reference velo vs dist plot
        p0.line(df_r.dist_lap, df_r.speedkmh, color=next(colors), line_width=2)

    # create an invisible renderer for velo vs dist
    # this is used to trigger the hover, thus the size is large
    c0 = p0.circle(x='dist_lap', y='speedkmh', source=ds, size=10, fill_alpha=0.0, alpha=0.0)
    c0.selection_glyph = Circle(fill_color='red', fill_alpha=1., line_color=None)
    c0.nonselection_glyph = Circle(fill_alpha=0, line_color=None)

    # create figure for track map
    p1 = figure(plot_height=400, plot_width=800, tools="crosshair,pan,reset,save,wheel_zoom")

    # create map of the track
    c1 = getLapFigure(p1, df_, ds, mode, hasref=(reference is not None))

    # add some lap tangents to guide the eye when comparing map and refmap
    if reference is not None and mode not in ['absolut', 'gainloss']:
        x0 = df_.x.values
        y0 = df_.y.values
        h = df_.heading.values
        x1 = x0 + 30*np.cos(h+np.pi/2)
        y1 = y0 + 30*np.sin(h+np.pi/2)
        p1.segment(x0=x0, y0=y0, x1=x1, y1=y1, color="#F4A582", line_width=1)

        # calculate points for the reference map drawn 'outside' of the other track map
        getLapFigure(p1, df_, ds , mode, ref=True)

    # Toooltips that show some information for each point, triggered via slider.onchange JS
    tools = ['time','dist','speedkmh']
    if reference is not None:
        tools.append('speedkmh_r')
        if mode not in ['absolut', 'gainloss', 'pedals', 'speed']:
            tools.extend([mode, '%s_r'%mode])
        elif mode in ['pedals']:
            tools.extend(['throttle', 'throttle_r', 'brake', 'brake_r'])
        tools.append('dt')
    elif mode == 'pedals':
        tools.extend(['throttle', 'brake'])
    elif mode != 'speed':
        tools.append(mode)

    hover0 = createHoverTool(tools)
    # a small hack to show only one tooltip (hover selects multiple points)
    hover0.tooltips[-1] = (hover0.tooltips[-1][0], hover0.tooltips[-1][1]+"""
        <style>
            .bk-tooltip>div:not(:first-child) {display:none;}
        </style>""")
    hover0.renderers = [r0]
    hover0.mode = 'vline'
    hover0.line_policy='interp'

    # selection change via button and slider. Tooltips 'hover0' will be rendered in 'p0' using rederer 'r0'
    slider = getLapSlider(ds, p0, r0, hover0, view=view)
    btns = getLapControls(ds, slider)

    # Hovertools, that emit a selection change by modifying the slider value
    callback = CustomJS(args=dict(slider=slider), code=
    """
    let val = cb_data['index'].indices[0]
    if (val!=0 && !isNaN(val))
        slider.value = cb_data['index'].indices[0];
    """)
    p0.add_tools(HoverTool(tooltips=None, renderers=[c0],
                           callback=callback,
                           line_policy='interp', mode='vline'))
    p1.add_tools(HoverTool(tooltips=None, renderers=[c1],
                           callback=callback,
                           line_policy='interp', mode='mouse', point_policy='snap_to_data'))

    p0.add_tools(hover0)
    # p1.add_tools(hover1)

    return column(btns, slider, p0, p1)




def getTrackMapPanel(df):
    mode_select = Select(title="Mode:", value='speed',
                         options=['oversteer',
                                  'speed',
                                  'pedals',
                                  'throttle',
                                  'brake',
                                  'g_lon'])

    layout = column(mode_select, getTrackMap(df, view='trackmap'))

    def mode_change(attrname, old, new):
        layout.children[1] = getTrackMap(df, view='trackmap', mode=new)

    mode_select.on_change('value', mode_change)

    return layout