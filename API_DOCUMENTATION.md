# RFdiffusion REST API Documentation

This document describes the REST API for RFdiffusion protein design service.

## Building and Running the Docker Container

### Build the Docker image:
```bash
cd RFdiffusion
docker build -f docker/Dockerfile -t rfdiffusion-api .
```

### Prepare directories and download models:
```bash
mkdir -p $HOME/inputs $HOME/outputs $HOME/models
bash scripts/download_models.sh $HOME/models
```

### Run the API server:
```bash
docker run -d --rm --gpus all \
  -p 8000:8000 \
  -v $HOME/models:/app/models \
  -v $HOME/inputs:/app/inputs \
  -v $HOME/outputs:/app/outputs \
  --name rfdiffusion-api \
  rfdiffusion-api
```

### Check logs:
```bash
docker logs -f rfdiffusion-api
```

### Stop the server:
```bash
docker stop rfdiffusion-api
```

## API Endpoints

### 1. Health Check

Check if the API server is running and view system information.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "output_dir": "/app/outputs",
  "model_dir": "/app/models"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Run Inference

Submit a protein design job.

**Endpoint:** `POST /api/v1/inference`

**Headers:**
- `Content-Type: application/json`

**Request Body:**
```json
{
  "config_name": "base",
  "inference.num_designs": 3,
  "inference.input_pdb": "/app/inputs/5TPN.pdb",
  "contigmap.contigs": ["10-40/A163-181/10-40"]
}
```

**Configuration Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `config_name` | string | Configuration file to use (`base`, `symmetry`) | `base` |
| `inference.num_designs` | integer | Number of designs to generate | 1 |
| `inference.input_pdb` | string | Path to input PDB file (optional) | - |
| `inference.output_prefix` | string | Output file prefix | auto-generated |
| `inference.model_directory_path` | string | Path to model weights | `/app/models` |
| `inference.deterministic` | boolean | Use deterministic mode | false |
| `inference.final_step` | integer | Final diffusion step | 1 |
| `inference.write_trajectory` | boolean | Write trajectory PDBs | false |
| `contigmap.contigs` | list[string] | Contig specification for design | - |
| `diffuser.T` | integer | Number of diffusion steps | 50 |
| `ppi.hotspot_res` | list[string] | Hotspot residues for PPI design | - |

**Response:**
```json
{
  "job_id": "abc123ef",
  "status": "completed",
  "num_designs": 3,
  "results": [
    {
      "design_number": 0,
      "pdb_file": "/app/outputs/design_abc123ef_0.pdb",
      "trb_file": "/app/outputs/design_abc123ef_0.trb",
      "time_seconds": 245.67,
      "mean_plddt": 0.847
    },
    {
      "design_number": 1,
      "pdb_file": "/app/outputs/design_abc123ef_1.pdb",
      "trb_file": "/app/outputs/design_abc123ef_1.trb",
      "time_seconds": 238.12,
      "mean_plddt": 0.852
    },
    {
      "design_number": 2,
      "pdb_file": "/app/outputs/design_abc123ef_2.pdb",
      "trb_file": "/app/outputs/design_abc123ef_2.trb",
      "time_seconds": 241.89,
      "mean_plddt": 0.839
    }
  ]
}
```

**Example - Unconditional Design:**
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 2,
    "contigmap.contigs": ["100-100"]
  }'
```

**Example - Motif Scaffolding:**
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 3,
    "inference.input_pdb": "/app/inputs/5TPN.pdb",
    "contigmap.contigs": ["10-40/A163-181/10-40"]
  }'
```

**Example - PPI Design:**
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 5,
    "inference.input_pdb": "/app/inputs/insulin_target.pdb",
    "contigmap.contigs": ["A1-150/0 B1-100"],
    "ppi.hotspot_res": ["A30", "A33", "A34"]
  }'
```

---

### 3. List Designs

List all generated design files.

**Endpoint:** `GET /api/v1/designs`

**Response:**
```json
{
  "total": 5,
  "designs": [
    {
      "pdb_file": "design_abc123ef_0.pdb",
      "has_trb": true,
      "size_bytes": 45678,
      "modified_time": 1698172345.123
    },
    {
      "pdb_file": "design_abc123ef_1.pdb",
      "has_trb": true,
      "size_bytes": 46012,
      "modified_time": 1698172590.456
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/designs
```

---

### 4. Download Files

Download generated PDB or TRB files.

**Endpoint:** `GET /api/v1/download/<filename>`

**Parameters:**
- `filename`: Relative path to file in output directory

**Response:** Binary file content

**Example - Download PDB:**
```bash
curl -O http://localhost:8000/api/v1/download/design_abc123ef_0.pdb
```

**Example - Download TRB (metadata):**
```bash
curl -O http://localhost:8000/api/v1/download/design_abc123ef_0.trb
```

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message description",
  "status": "failed"
}
```

Common HTTP status codes:
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error during processing

---

## Environment Variables

The following environment variables can be set when running the container:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API server port | `8000` |
| `HOST` | API server host | `0.0.0.0` |
| `OUTPUT_DIR` | Directory for output files | `/app/outputs` |
| `MODEL_DIR` | Directory containing model weights | `/app/models` |

**Example with custom environment:**
```bash
docker run -d --rm --gpus all \
  -p 9000:9000 \
  -e PORT=9000 \
  -e OUTPUT_DIR=/app/my_outputs \
  -v $HOME/models:/app/models \
  -v $HOME/outputs:/app/my_outputs \
  --name rfdiffusion-api \
  rfdiffusion-api
```

---

## Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Submit inference job
config = {
    "inference.num_designs": 2,
    "contigmap.contigs": ["50-50"],
    "inference.deterministic": True
}

response = requests.post(
    f"{BASE_URL}/api/v1/inference",
    json=config,
    headers={"Content-Type": "application/json"}
)

result = response.json()
print(f"Job ID: {result['job_id']}")
print(f"Status: {result['status']}")

# Download first design
if result['status'] == 'completed':
    pdb_file = result['results'][0]['pdb_file']
    filename = pdb_file.split('/')[-1]
    
    response = requests.get(f"{BASE_URL}/api/v1/download/{filename}")
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded: {filename}")
```

---

## Notes

1. **GPU Memory**: Each inference job requires significant GPU memory. Running multiple concurrent jobs may cause out-of-memory errors.

2. **File Persistence**: Output files are stored in the mounted volume. Make sure to persist the `/app/outputs` directory.

3. **Model Files**: Ensure model weights are properly downloaded and mounted before starting the server.

4. **Trajectory Files**: Setting `inference.write_trajectory=true` will generate large trajectory PDB files in a `traj/` subdirectory.

5. **Input Files**: When specifying `inference.input_pdb`, ensure the file is accessible within the container (typically in `/app/inputs`).

## Support

For issues and questions:
- Original RFdiffusion: https://github.com/RosettaCommons/RFdiffusion
- Report API-specific issues in your deployment repository
