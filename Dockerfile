FROM ubuntu:latest

LABEL maintainer="latruonghai@gmail.com"

WORKDIR /app



COPY . .
RUN cd /app/lib/Flask && make deploy
CMD echo done

