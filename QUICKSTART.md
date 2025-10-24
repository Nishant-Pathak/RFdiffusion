# RFdiffusion REST API - Quick Start Guide

This guide will help you quickly get the RFdiffusion REST API up and running.

## Prerequisites

- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA support
- At least 16GB GPU memory (recommended)
- Sufficient disk space for models (~2GB) and outputs

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion
```

### 2. Prepare Directories

```bash
mkdir -p $HOME/models $HOME/inputs $HOME/outputs
```

### 3. Download Model Weights

```bash
bash scripts/download_models.sh $HOME/models
```

This will download the pre-trained RFdiffusion models (~2GB).

### 4. Build the Docker Image

```bash
docker build -f docker/Dockerfile -t rfdiffusion-api .
```

This may take 10-15 minutes to complete.

### 5. Start the API Server

#### Option A: Using Docker Run

```bash
docker run -d --rm --gpus all \
  -p 8000:8000 \
  -v $HOME/models:/app/models \
  -v $HOME/inputs:/app/inputs \
  -v $HOME/outputs:/app/outputs \
  --name rfdiffusion-api \
  rfdiffusion-api
```

#### Option B: Using Docker Compose

```bash
docker-compose up -d
```

### 6. Verify the Server is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GPU Name",
  "output_dir": "/app/outputs",
  "model_dir": "/app/models"
}
```

## Basic Usage

### Example 1: Simple Unconditional Design

Generate a 100-residue protein:

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 1,
    "contigmap.contigs": ["100-100"]
  }'
```

### Example 2: Motif Scaffolding

First, add an input PDB to the inputs directory:

```bash
wget -P $HOME/inputs https://files.rcsb.org/view/5TPN.pdb
```

Then run motif scaffolding:

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 3,
    "inference.input_pdb": "/app/inputs/5TPN.pdb",
    "contigmap.contigs": ["10-40/A163-181/10-40"]
  }'
```

### Example 3: List Generated Designs

```bash
curl http://localhost:8000/api/v1/designs
```

### Example 4: Download a Design

```bash
curl -O http://localhost:8000/api/v1/download/design_abc123ef_0.pdb
```

## Testing the API

Run the included test script:

```bash
# Install requests library
pip install requests

# Run quick test (skips time-consuming inference)
python scripts/test_api.py --skip-inference

# Run full test including inference
python scripts/test_api.py --num-designs 1
```

## Monitoring

### View Logs

```bash
docker logs -f rfdiffusion-api
```

### Check GPU Usage

```bash
nvidia-smi
```

### Monitor Container Resources

```bash
docker stats rfdiffusion-api
```

## Managing the Server

### Stop the Server

```bash
docker stop rfdiffusion-api
```

Or with docker-compose:
```bash
docker-compose down
```

### Restart the Server

```bash
docker restart rfdiffusion-api
```

Or with docker-compose:
```bash
docker-compose restart
```

### View Container Status

```bash
docker ps
```

## Common Issues

### Issue: "No GPU detected"

**Solution:** Ensure nvidia-docker2 is installed and GPU is accessible:
```bash
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

### Issue: "Model files not found"

**Solution:** Verify models are downloaded and mounted correctly:
```bash
ls -la $HOME/models
docker exec rfdiffusion-api ls -la /app/models
```

### Issue: "Permission denied" when writing outputs

**Solution:** Ensure output directory has correct permissions:
```bash
chmod -R 777 $HOME/outputs
```

### Issue: Port 8000 already in use

**Solution:** Use a different port:
```bash
docker run -d --rm --gpus all \
  -p 9000:8000 \
  -v $HOME/models:/app/models \
  -v $HOME/outputs:/app/outputs \
  --name rfdiffusion-api \
  rfdiffusion-api
```

Then access at `http://localhost:9000`

## Next Steps

- Read the full [API Documentation](API_DOCUMENTATION.md)
- Explore design examples in the `examples/` directory
- Check the original [RFdiffusion README](README.md) for more details on parameters

## Performance Tips

1. **Reduce diffusion steps for faster testing:**
   ```json
   {"diffuser.T": 10}
   ```
   Default is 50 steps. Lower values are faster but may reduce quality.

2. **Use deterministic mode for reproducibility:**
   ```json
   {"inference.deterministic": true}
   ```

3. **Batch multiple designs in one request:**
   ```json
   {"inference.num_designs": 10}
   ```

4. **Monitor GPU memory:**
   Large proteins or many concurrent requests may cause OOM errors.

## Python Client Example

```python
import requests

# Submit job
response = requests.post(
    "http://localhost:8000/api/v1/inference",
    json={
        "inference.num_designs": 2,
        "contigmap.contigs": ["80-80"]
    }
)

result = response.json()
print(f"Job completed: {result['job_id']}")

# Download first design
for design in result['results']:
    filename = design['pdb_file'].split('/')[-1]
    pdb_data = requests.get(
        f"http://localhost:8000/api/v1/download/{filename}"
    ).content
    
    with open(filename, 'wb') as f:
        f.write(pdb_data)
    print(f"Downloaded: {filename}")
```

## Support

For issues and questions:
- RFdiffusion GitHub: https://github.com/RosettaCommons/RFdiffusion
- API Documentation: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
