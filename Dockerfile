#TODO: check what the pytorch image has installed and maybe use that one

FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
LABEL MANTAINER="Erick Cobos <ecobos@tuebingen.mpg.de>"
WORKDIR /src

# Upgrade system 
RUN apt update && apt upgrade -y

# Install dependencies
RUN pip install pandas scikit-learn h5py

# Install dermosxai
ADD ./setup.py /src/dermosxai/setup.py
ADD ./dermosxai /src/dermosxai/dermosxai
RUN pip install -e /src/dermosxai

# Install extra libraries (non-essential but useful)
RUN apt install -y python3-tk nano
RUN pip install matplotlib jupyterlab #ipympl
# https://github.com/jupyterlab/jupyterlab/issues/9226#issuecomment-716261418 : according to this comment in jupyterlab v3.0, interactive matplotlib will only require installing ipympl

# Clean apt lists
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
