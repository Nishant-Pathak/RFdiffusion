# Usage (REST API): 
# git clone https://github.com/RosettaCommons/RFdiffusion.git
# cd RFdiffusion
# docker build -f docker/Dockerfile -t rfdiffusion-api .
# mkdir -p $HOME/inputs $HOME/outputs $HOME/models
# bash scripts/download_models.sh $HOME/models
#
# Run API Server:
# docker run -d --rm --gpus all \
#   -p 8000:8000 \
#   -v $HOME/models:/app/models \
#   -v $HOME/inputs:/app/inputs \
#   -v $HOME/outputs:/app/outputs \
#   --name rfdiffusion-api \
#   rfdiffusion-api
#
# Test the API:
# curl http://localhost:8000/health
#
# Submit inference job:
# curl -X POST http://localhost:8000/api/v1/inference \
#   -H "Content-Type: application/json" \
#   -d '{"inference.num_designs": 1, "contigmap.contigs": ["100-100"]}'
#
# See API_DOCUMENTATION.md for full API reference
#
# To run original CLI mode instead:
# docker run -it --rm --gpus all \
#   -v $HOME/models:/app/models \
#   -v $HOME/inputs:/app/inputs \
#   -v $HOME/outputs:/app/outputs \
#   --entrypoint python3.9 \
#   rfdiffusion-api \
#   scripts/run_inference.py \
#   inference.output_prefix=/app/outputs/motifscaffolding \
#   inference.model_directory_path=/app/models \
#   inference.num_designs=3 \
#   'contigmap.contigs=[100-100]'

FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

COPY . /app/RFdiffusion/

RUN apt-get -q update \ 
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  git \
  curl \
  wget \
  software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt-get update \
  && apt-get install --no-install-recommends -y \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean \
  && pip install -q --no-cache-dir \
  numpy==1.23.5 \
  && pip install -q --no-cache-dir \
  torch==1.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html \
  && pip install -q --no-cache-dir \
  dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
  && pip install -q --no-cache-dir \
  e3nn==0.3.3 \
  wandb==0.12.0 \
  pynvml==11.0.0 \
  git+https://github.com/NVIDIA/dllogger#egg=dllogger \
  decorator==5.1.0 \
  hydra-core==1.3.2 \
  pyrsistent==0.19.3 \
  flask==2.3.3 \
  sqlalchemy==2.0.24 \
  psycopg2-binary==2.9.11 \
  flask-cors==4.0.0 \
  /app/RFdiffusion/env/SE3Transformer \
  && pip install --no-cache-dir /app/RFdiffusion --no-deps
  
WORKDIR /app/RFdiffusion

ENV DGLBACKEND="pytorch"
ENV OUTPUT_DIR="/app/outputs"
ENV MODEL_DIR="/app/models"
ENV PORT=8000

# Create output directory
RUN mkdir -p /app/outputs /app/models

# Expose API port
EXPOSE 8000

ENTRYPOINT ["python3.9", "scripts/api_server.py"]
