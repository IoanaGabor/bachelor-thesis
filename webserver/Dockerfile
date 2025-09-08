FROM python:3.13.1

WORKDIR /webserver

COPY . /webserver

RUN apt-get update && apt-get install -y libpq-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

RUN chmod +x ./app/start.sh

EXPOSE 8080

ENV APP_MODULE="main:app"
ENV APP_HOST="0.0.0.0"
ENV APP_PORT="8080"

CMD ["./app/start.sh"]
