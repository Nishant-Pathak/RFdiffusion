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

# Install system dependencies (cached layer)
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
  && apt-get clean

# Copy requirements first (for better caching)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies (cached layer)
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
  && pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Copy SE3Transformer and install
COPY env/SE3Transformer /tmp/SE3Transformer
RUN pip install --no-cache-dir /tmp/SE3Transformer

# Copy application code
COPY . /app/RFdiffusion/

# Install RFdiffusion
RUN pip install --no-cache-dir /app/RFdiffusion --no-deps
  
WORKDIR /app/RFdiffusion

ENV DGLBACKEND="pytorch"
ENV OUTPUT_DIR="/app/results"
ENV MODEL_DIR="/app/models"
ENV PORT=8080
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/RFdiffusion/google-conf.json"


# Create output directory
RUN mkdir -p /app/results /app/models

# Expose API port
EXPOSE 8080

ENTRYPOINT ["python3.9", "scripts/api_server.py"]
