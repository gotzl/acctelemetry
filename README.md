An application to display telemetry data recorded by Assetto Corsa Competizione.

It is build around bokeh and displays various figures that are inspired by the ACC MoTec i2 workspace.
See also [this article](https://www.racedepartment.com/threads/acc-blog-motec-telemetry-and-dedicated-acc-workspace.165714/). 

The folder for the telemetry files is set via an env variable. All ld files in that folder are listed in a table, split up lap-by-lap.
One or more laps can be selected and the data is displayed in the various tabs after hitting the 'Load' button.


## Dependencies
```bash
pip3 install -r requirements.txt
```

## Usage
```bash
export TELEMETRY_FOLDER=='/../Documents/Assetto Corsa Competizione/MoTeC'
bokeh serve --show ../acctelemetry
```

There is also a docker image
```bash
docker build -t acctelemetry .
docker run --name acctelemetry -p 5100:5100 -e ORIGIN=www.example.com:5100 -d --rm acctelemetry
```