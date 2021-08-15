#!/bin/sh
export PYTHONPATH=/acctelemetry/:$PYTHONPATH
if [ -z "${PREFIX}" ]; then
    PREFIX_PARAM="";
else
    PREFIX_PARAM="--prefix ${PREFIX}";
fi
/root/.local/bin/bokeh serve --show /acctelemetry\
    --websocket-max-message-size 200000000\
    --port ${PORT}\
    --address 0.0.0.0\
    --allow-websocket-origin=${ORIGIN}\
    --log-level ${LOG_LEVEL}\
    --use-xheaders\
    ${PREFIX_PARAM}
