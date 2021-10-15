FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
LABEL MANTAINER="Erick Cobos <ecobos@tuebingen.mpg.de>"
WORKDIR /src

# Upgrade system 
RUN apt update && apt upgrade -y

# Install dependencies
RUN pip install pandas scikit-learn h5py wandb

# Install dermosxai
ADD ./setup.py /src/dermosxai/setup.py
ADD ./dermosxai /src/dermosxai/dermosxai
RUN pip install -e /src/dermosxai

# Install extra libraries (non-essential but useful)
RUN apt install -y python3-tk nano
RUN pip install matplotlib jupyterlab ipympl seaborn pydicom
RUN pip uninstall jedi -y # jupyter autocompletions are broken with this: https://stackoverflow.com/questions/40536560/ipython-and-jupyter-autocomplete-not-working

# Clean apt lists
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
