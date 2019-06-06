import itertools
from bokeh.palettes import Spectral4, Dark2_5 as palette
from bokeh.plotting import figure
from bokeh.models import CrosshairTool, HoverTool, CustomJS,\
    LinearAxis, ColumnDataSource, Range1d
from bokeh.layouts import gridplot, column


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


def getSimpleFigure(df, vars, tools, extra_y=None, extra_y_vars=None):
    WIDTH = 800
    TOOLS = "crosshair,pan,reset,save,wheel_zoom"

    # create a new plot with a title and axis labels
    p = figure(plot_height=400, plot_width=WIDTH, tools=TOOLS,
               x_axis_label='Dist [m]')

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
    vars = ['speedkmh','steerangle','g_lat', 'oversteer']
    tools = ['time','dist']+vars
    return getSimpleFigure(df, vars+['tc', 'throttle','brake'], tools,
                           {"pedals": Range1d(start=-20, end=400), "tc": Range1d(start=-1, end=20), "oversteer": Range1d(start=-15, end=25)},
                           {"pedals":['throttle','brake'], "tc":['tc'], "oversteer":['oversteer','g_lat']})

