FROM ubuntu:16.04

WORKDIR /main

ADD . /main

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev


# update pip
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install wheel


RUN pip install -r requirements.txt

EXPOSE 8080

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

COPY . /main

ENTRYPOINT [ "python3" ]

CMD ["main.py" ]
