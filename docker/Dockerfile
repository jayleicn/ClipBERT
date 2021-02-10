FROM nvcr.io/nvidia/pytorch:19.10-py3
COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/NVIDIA/apex.git &&\
    cd apex &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    rm -rf ../apex

#### detectron 2 installation
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'
################## v1 ##########################

####### horovod for multi-GPU (distributed) training #######

# update OpenMPI to avoid horovod bug
RUN rm -r /usr/local/mpi &&\ 
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz &&\
    gunzip -c openmpi-4.0.0.tar.gz | tar xf - &&\
    cd openmpi-4.0.0 &&\
    ./configure --prefix=/usr/local/mpi --enable-orterun-prefix-by-default &&\
    make -j$(nproc) all && make install &&\
    ldconfig &&\
    cd - && rm -r openmpi-4.0.0 && rm openmpi-4.0.0.tar.gz

ENV OPENMPI_VERSION=4.0.0

# horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==0.19.4 &&\
    ldconfig

# ssh
RUN apt-get update &&\
    apt-get install -y --no-install-recommends openssh-client openssh-server &&\
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# fix ssh permissions
RUN bash -c "chmod -R 600 /etc/ssh/ && chmod 600 /var/run/sshd/ && chmod 600 /root"

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
RUN pip uninstall pillow && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

WORKDIR /clipbert
