FROM ubuntu:latest

WORKDIR /app

COPY dist/SurfEncoder /app/

CMD ["./app/SurfEncoder"]
