FROM nvcr.io/nvidia/pytorch:20.10-py3
RUN apt-get update
COPY . /emphysema_seg
RUN pip install /emphysema_seg