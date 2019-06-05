# -*- coding: utf-8 -*-
"""
https://github.com/bokeh/bokeh/issues/6096#issuecomment-299002827
Created on Wed May 03 11:26:21 2017

@author: Kevin Anderson
"""

def uploadButton(base, data_table):
    from bokeh.models import ColumnDataSource, CustomJS
    from bokeh.models.widgets import Button
    from acctelemetry import createSource

    import os, base64, glob

    file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})

    def file_callback(attr,old,new):
        print('filename:', file_source.data['file_name'])
        raw_contents = file_source.data['file_contents'][0]
        # remove the prefix that JS adds
        prefix, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents)
        f = os.path.join(base, file_source.data['file_name'][0])
        if not os.path.exists(f):
            with open(f, 'wb') as f_:
                f_.write(file_contents)

            global source
            files = glob.glob(base+'/*.ld')
            source = createSource(files)
            data_table.source = source

    file_source.on_change('data', file_callback)

    button = Button(label="Upload", button_type="success")
    button.callback = CustomJS(args=dict(file_source=file_source), code = """
    function read_file(filename) {
        var reader = new FileReader();
        reader.onload = load_handler;
        reader.onerror = error_handler;
        // readAsDataURL represents the file's data as a base64 encoded string
        reader.readAsDataURL(filename);
    }
    
    function load_handler(event) {
        var b64string = event.target.result;
        file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
        file_source.trigger("change");
    }
    
    function error_handler(evt) {
        if(evt.target.error.name == "NotReadableError") {
            alert("Can't read file!");
        }
    }
    
    var input = document.createElement('input');
    input.setAttribute('type', 'file');
    input.onchange = function(){
        if (window.FileReader) {
            read_file(input.files[0]);
        } else {
            alert('FileReader is not supported in this browser');
        }
    }
    input.click();
    """)
    return button