FROM python:3.6-slim-stretch

ENV TELEMETRY_FOLDER=/telemetry
RUN mkdir /telemetry

ADD requirements.txt .
RUN pip3 install --user --no-cache-dir -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ORIGIN="localhost:5100" PORT="5100" PREFIX="" LOG_LEVEL="debug"

ADD . /acctelemetry

ENTRYPOINT ["./entrypoint.sh"]
