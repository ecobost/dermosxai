FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
LABEL MANTAINER="Erick Cobos <ecobos@tuebingen.mpg.de>"
WORKDIR /src

# Upgrade system 
RUN apt update && apt upgrade -y

# Install dependencies
RUN apt install -y build-essential && \
    pip install scikit-learn h5py wandb shap

# Install dermosxai
ADD ./setup.py /src/dermosxai/setup.py
ADD ./dermosxai /src/dermosxai/dermosxai
RUN pip install -e /src/dermosxai

# Install extra libraries (non-essential but useful)
RUN apt install -y python3-tk nano
RUN pip install matplotlib jupyterlab ipympl seaborn pydicom pandas scikit-image

# Clean apt lists
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
