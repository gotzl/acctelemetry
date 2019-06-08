import os, itertools
import numpy as np
from bokeh.io import curdoc

from bokeh.palettes import Spectral4, Dark2_5 as palette
from bokeh.plotting import figure
from bokeh.models import CrosshairTool, HoverTool, CustomJS, \
    LinearAxis, ColumnDataSource, Range1d, TextInput, Circle, Select, Line, Button, Selection, Slider, TapTool, LabelSet
from bokeh.layouts import gridplot, column, row

import acctelemetry, laptable
from ldparser import ldparser

def createHoverTool(tools, mode='mouse'):
    _tools = dict(
        dist=("Dist", "@dist_lap{%0.1f} m"),
        time=("Time", "@time_lap{%0.1f} s"),
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
                'dist' : 'printf',
                'dist_lap' : 'printf',
                'time_lap' : 'printf',
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
                           {"sus": Range1d(start=-10, end=120)})


def getWheelSpeedFigure(df):
    vars = ['wheel_speed_lf','wheel_speed_lr',
            'wheel_speed_rf', 'wheel_speed_rr',
            'throttle','brake']
    tools = ['time','dist','speed']+vars
    return getSimpleFigure(df, vars+['tc', 'abs'], tools,
                           {"pedals": Range1d(start=-20, end=400), "tcabs": Range1d(start=-1, end=20)},
                           {"pedals":['throttle','brake'], "tcabs":['tc', 'abs']})


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
        callback("absolut")

    def callback(mode):
        idxs = filter_source.selected.indices
        if (len(idxs)<2):
            fig.children[0] = tmp
            return

        df, track, reference, target = None, None, None, None
        for idx in idxs:
            name = filter_source.data['name'][idx]
            f_ = os.environ['TELEMETRY_FOLDER']+'/%s'%name
            head_, chans = ldparser.read_ldfile(f_)

            # if track is not None and head_.descr1!=track: continue

            laps = acctelemetry.laps(f_)
            laps_limits = acctelemetry.laps_limits(laps, chans[4].freq, len(chans[4].data))

            laps = np.array(laps)
            laps_times = [laps[0]]
            laps_times.extend(list(laps[1:]-laps[:-1]))

            # create pandas DataFrame
            df_ = acctelemetry.createDataFrame(
                name, chans, laps_times, laps_limits)
            # restrict to selected lap
            lap = int(filter_source.data['lap'][idx])
            df_ = df_[df_.lap==lap]


            info = [name, lap, head_.descr2, laps_times[lap]]
            if df is None:
                df = df_
                reference = info
                track = head_.descr1
            else:
                df = df.append(df_)
                target = info

        if reference is None or target is None:
            fig.children[0] = tmp
            return

        text_input.value = "%s: reference: %s (%i) | target: %s (%i)"%(track, reference,idxs[0],target,idxs[-1])
        text_input.value = "%s | %s: reference: %s / %.3f (%i) | target: %s / %.3f (%i)"%\
                           (track, mode, reference[2], reference[3], idxs[0],
                            target[2], target[3], idxs[-1])

        fig.children[0] = getLapDeltaFigure(df, reference[:2], target[:2], mode)

    def mode_change(attrname, old, new):
        # if (old==new) or p1 is None: return
        callback(new)
        # c = 'color_absolut'
        # if new == 'gainloss':
        #     c = 'color_grad'
        # global ds
        # ds.data['color'] = ds.data[c]
        # ds.trigger('indices', None, None)

    text_input = TextInput(value="nothing selected")
    text_input.disabled = True

    mode_select = Select(title="Mode:", value='absolut',
                         options=['absolut',
                                  'gainloss',
                                  'speed',
                                  'pedals',
                                  'g_lon'])
    mode_select.on_change('value', mode_change)

    filter_source.selected.on_change('indices', callback_)
    tmp = figure(plot_height=500, plot_width=800)
    fig = row(tmp)
    return column(filters,data_table,mode_select,text_input,fig)


def getLapDeltaFigure(df, reference, target, mode='absolut'):
    df_, df_r = acctelemetry.lapdelta(df, reference, target)
    color_absolut = acctelemetry.deltacolors(df_.dt.values)
    color_gainloss = acctelemetry.deltacolors(df_.dt.values, style='grad')

    # add colors to dataframe
    df_ = acctelemetry.addColorMaps(df_, {'color_absolut':color_absolut,
                                          'color_gainloss':color_gainloss})

    p0 = figure(plot_height=400, plot_width=800,
                tools="crosshair,pan,reset,save,wheel_zoom")

    ds = ColumnDataSource(df_)
    colors = itertools.cycle(palette)
    c0,c1 = next(colors),next(colors)

    # create the velo vs dist plot
    r0 = p0.line(x='dist_lap', y='speedkmh', source=ds, color=c0, line_width=2)
    # overwrite the (non)selection glyphs with the base line style
    # the style for the hover will be set below
    nonselected_ = Line(line_alpha=1, line_color=c0, line_width=2)
    r0.selection_glyph = nonselected_
    r0.nonselection_glyph = nonselected_

    # create the dt vs dist plot with extra y axis, set the (non)selection glyphs
    lim = max(df_.dt.abs())
    lim += lim*.2
    p0.extra_y_ranges = {"dt": Range1d(start=-lim, end=lim)}
    p0.add_layout(LinearAxis(
        y_range_name='dt',
        axis_label='dt [s]'), 'right')
    r1 = p0.line(x='dist_lap', y='dt', source=ds, y_range_name='dt', color=c1, line_width=2)
    r1.selection_glyph = Line(line_alpha=1, line_color='red', line_width=5)
    r1.nonselection_glyph = Line(line_alpha=1, line_color=c1, line_width=2)

    # create reference velo vs dist plot
    p0.line(df_r.dist_lap, df_r.speedkmh, color=next(colors), line_width=2)

    # create second figure for track map
    p1 = figure(plot_height=400, plot_width=800, tools="crosshair,pan,reset,save,wheel_zoom")
    # plot the track map, overwrite the (non)selection glyph to keep our color from ds
    # the hover effect is configured below
    r2 = p1.scatter(x='x', y='y', source=ds, color='color_%s'%mode)
    r2.nonselection_glyph = r2.selection_glyph

    # calculate points for the reference map drawn 'outside' of the other track map
    if mode not in ['absolut', 'gainloss']:
        ds.data['xr'] = df_.xr+30*np.cos(df_.heading+np.pi/2)
        ds.data['yr'] = df_.yr+30*np.sin(df_.heading+np.pi/2)
        r3 = p1.scatter(x='xr', y='yr', source=ds, color='color_%s_r'%mode)
        r3.nonselection_glyph = r3.selection_glyph

    # create a invisible renderer for velo vs dist
    # this is used to trigger the hover, thus the size is large
    c0 = p0.circle(x='dist_lap', y='speedkmh', source=ds, size=10, fill_alpha=0.0, alpha=0.0)
    c0.selection_glyph = Circle(fill_color='red', fill_alpha=1., line_color=None)
    c0.nonselection_glyph = Circle(fill_alpha=0, line_color=None)

    # create a invisible renderer for the track map
    # this is used to trigger the hover, thus the size is large
    c1 = p1.circle(x='x', y='y', source=ds, size=10, fill_alpha=0.0, alpha=0.0)
    c1.selection_glyph = Circle(fill_color='red', fill_alpha=.7, line_color=None)
    c1.nonselection_glyph = Circle(fill_alpha=0, line_color=None)

    cr = p1.circle(x='xr', y='yr', source=ds,
                   size = 8 if mode in ['absolut', 'gainloss'] else 10,
                   fill_alpha=0.0, alpha=0.0)
    cr.selection_glyph = Circle(fill_color='blue', fill_alpha=.7, line_color=None)
    cr.nonselection_glyph = Circle(fill_alpha=0, line_color=None)


    # Update the selection with slider.
    # Selection changes from slider trigger the display of a label next
    # to the selected point.
    slider = Slider(start=0, end=len(ds.data['dist_lap']),value=0, step=50)
    code = """
    labels.data = {'x':[],'y':[],'t':[]}
    if (cb_data && cb_data['index']) {
        cb_data['index'].indices = [cb_data['index'].indices[0]];
        source.selected = cb_data['index'];
    } else {
        source.selected.indices = [slider.value]
        labels.data = {'ind':[slider.value],
                'x':[source.data.dist_lap[slider.value]],
                'y':[source.data.speedkmh[slider.value]],
                't':[source.data.speedkmh[slider.value]]}
    }
    labels.change.emit()
    source.change.emit()
    """
    labels = ColumnDataSource(data=dict(x=[], y=[], t=[], ind=[]))
    p0.add_layout(LabelSet(x='x', y='y', text='t', y_offset=10, x_offset=10, source=labels))
    callback = CustomJS(args=dict(source=ds, labels=labels, slider=slider), code=code)


    # A tooltip that shows some information for each point,
    # synchronized to the track map via the callback
    hover0 = createHoverTool(['time','dist','speed','dt'])
    # a small hack to show only one tooltip (hover selects multiple points)
    hover0.tooltips[-1] = (hover0.tooltips[-1][0], hover0.tooltips[-1][1]+"""
        <style>
            .bk-tooltip>div:not(:first-child) {display:none;}
        </style>""")
    hover0.renderers=[c0]
    hover0.callback = callback
    hover0.point_policy='snap_to_data'
    hover0.mode = 'vline'
    p0.add_tools(hover0)

    # A tooltip that shows some information for each point,
    # synchronized to the track map via the callback
    hover = createHoverTool(['time','dist','speed','dt'])
    hover.tooltips = hover0.tooltips
    hover.renderers=[c1]
    hover.callback = callback
    hover.point_policy='snap_to_data'
    p1.add_tools(hover)


    #### create a player that iterates over the data
    def increment(stepsize, direction=1):
        idxs = ds.selected.indices
        if idxs is None or \
                len(idxs)==0 or \
                idxs[0] is None:
            i = 0
        else:
            i = idxs[0] + direction*stepsize

        if i<0: i = 0
        if i>len(ds.data['dist']): i = len(ds.data['dist'])-1
        # set the slider value, this will in turn trigger the
        # update of the selected item in the ds
        if slider.value == i:
            ds.selected = Selection(indices=[i])
            ds.trigger('data', ds.data, ds.data)
        else: slider.value = i

    def cbcrl(stop=False):
        global cb

        if cb is None and not stop:
            cb = curdoc().add_periodic_callback(lambda: increment(5), 5*50)
            # reset hovertool
            hover0.renderers = []
            hover.renderers = []
            play.label = "Pause"
        else:
            try:  curdoc().remove_periodic_callback(cb)
            except: pass
            cb = None
            play.label = "Play"
            if stop:
                # reset hovertool and selection
                ds.selected = Selection(indices=[])
                ds.trigger('data', ds.data, ds.data)
                hover0.renderers = [c0]
                hover.renderers = [c1]

    def stopcb():
        cbcrl(stop=True)

    global cb
    if not 'cb' in globals():
        cb = None
    if cb is not None:
        try:  curdoc().remove_periodic_callback(cb)
        except: cb = None

    play = Button(label="Play")
    play.on_click(cbcrl)

    stop = Button(label="Stop")
    stop.on_click(stopcb)

    btns = []
    def bb(): increment(50, -1)
    def b(): increment(5, -1)
    def f(): increment(5, 1)
    def ff(): increment(50, 1)
    for l,s in [('<<',bb),('<',b),('>',f),('>>',ff)]:
        b = Button(label=l, width=50)
        b.on_click(s)
        btns.append(b)

    slider.js_on_change('value', callback)
    return column(row(play, stop, *btns), slider, p0, p1)
