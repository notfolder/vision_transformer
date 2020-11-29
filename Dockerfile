FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 0 && \
    update-alternatives --set python /usr/bin/python3 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 && \
    update-alternatives --set pip /usr/bin/pip3 && \
    pip install --upgrade pip

#RUN pip install --upgrade jax==0.1.57 jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN apt-get install -y python3-dev g++ git
RUN pip install numpy==1.18.5 scipy six==1.15.0 wheel

RUN git clone https://github.com/google/jax && cd jax && python build/build.py --enable_cuda && pip install dist/*.whl

#COPY vit_jax/requirements.txt /tmp/
#RUN pip install -r tmp/requirements.txt

RUN apt-get install -y wget git
WORKDIR /root
RUN git clone https://github.com/google-research/vision_transformer.git
WORKDIR /root/vision_transformer
RUN pip install -r vit_jax/requirements.txt
#RUN wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

#CMD ["/bin/bash"]
COPY entrypoint.sh /root/vision_transformer/
RUN chmod +x /root/vision_transformer/entrypoint.sh
ENTRYPOINT ["/root/vision_transformer/entrypoint.sh"]
