# RFdiffusion REST API - Changes Summary

## Overview

The RFdiffusion inference script has been successfully converted to expose a REST API service. The original command-line interface is still available, and the Docker container now starts an API server by default.

## Files Created

### 1. `scripts/api_server.py` (New)
- **Purpose**: Flask-based REST API server for RFdiffusion inference
- **Features**:
  - Health check endpoint (`/health`)
  - Inference endpoint (`/api/v1/inference`)
  - File download endpoint (`/api/v1/download/<filename>`)
  - List designs endpoint (`/api/v1/designs`)
  - Full CORS support
  - Comprehensive error handling
  - Job tracking with unique IDs

### 2. `API_DOCUMENTATION.md` (New)
- **Purpose**: Complete API reference documentation
- **Contents**:
  - Endpoint descriptions and parameters
  - Request/response examples
  - Configuration options table
  - Python client examples
  - Common error codes
  - Environment variables reference

### 3. `QUICKSTART.md` (New)
- **Purpose**: Quick start guide for users
- **Contents**:
  - Step-by-step setup instructions
  - Basic usage examples
  - Troubleshooting guide
  - Performance tips
  - Docker commands reference

### 4. `scripts/test_api.py` (New)
- **Purpose**: Automated testing script for the API
- **Features**:
  - Health check testing
  - Inference endpoint testing
  - Download endpoint testing
  - List designs testing
  - Comprehensive test reporting

### 5. `examples/api_usage.sh` (New)
- **Purpose**: Example shell script demonstrating API usage
- **Contents**:
  - Curl command examples
  - Python client examples
  - Various design scenarios
  - Real working examples

### 6. `docker-compose.yml` (New)
- **Purpose**: Docker Compose configuration for easy deployment
- **Features**:
  - GPU support configuration
  - Volume mappings
  - Environment variables
  - Health checks
  - Auto-restart policy

## Files Modified

### 1. `docker/Dockerfile`
**Changes:**
- Added Flask dependencies (`flask==2.3.3`, `flask-cors==4.0.0`)
- Added `curl` for health checks
- Changed `ENTRYPOINT` from `run_inference.py` to `api_server.py`
- Added environment variables: `OUTPUT_DIR`, `MODEL_DIR`, `PORT`
- Added `EXPOSE 8000` directive
- Created output and model directories
- Updated usage comments with API examples

**Backward Compatibility:**
- Original CLI mode still available by overriding entrypoint:
  ```bash
  docker run --entrypoint python3.9 rfdiffusion-api scripts/run_inference.py [args]
  ```

## API Endpoints

### Health Check
- **URL**: `GET /health`
- **Purpose**: Verify server status and GPU availability
- **Response**: Server status, GPU info, directories

### Run Inference
- **URL**: `POST /api/v1/inference`
- **Purpose**: Submit protein design jobs
- **Input**: JSON configuration with Hydra parameters
- **Output**: Job results with PDB file paths and metadata

### List Designs
- **URL**: `GET /api/v1/designs`
- **Purpose**: List all generated design files
- **Output**: Array of design files with metadata

### Download Files
- **URL**: `GET /api/v1/download/<filename>`
- **Purpose**: Download generated PDB/TRB files
- **Output**: Binary file content

## Key Features

### 1. RESTful API
- Standard HTTP methods (GET, POST)
- JSON request/response format
- CORS enabled for cross-origin requests
- Proper HTTP status codes

### 2. Job Management
- Unique job IDs for tracking
- Automatic output file naming
- Metadata storage in TRB files
- Support for multiple concurrent jobs

### 3. Configuration Flexibility
- Full Hydra configuration support
- Override any parameter via JSON
- Support for all original RFdiffusion modes:
  - Unconditional design
  - Motif scaffolding
  - PPI design
  - Symmetric oligomers
  - Partial diffusion

### 4. Docker Integration
- Single-step deployment
- GPU support via nvidia-docker
- Volume mounting for models, inputs, outputs
- Environment variable configuration
- Health check integration

### 5. Error Handling
- Comprehensive exception catching
- Detailed error messages
- Proper HTTP error codes
- Request validation

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API server port |
| `HOST` | `0.0.0.0` | API server host |
| `OUTPUT_DIR` | `/app/outputs` | Output directory path |
| `MODEL_DIR` | `/app/models` | Model weights directory |
| `DGLBACKEND` | `pytorch` | DGL backend (required) |

## Usage Examples

### Quick Start
```bash
# Build and run
docker build -f docker/Dockerfile -t rfdiffusion-api .
docker run -d --gpus all -p 8000:8000 \
  -v $HOME/models:/app/models \
  -v $HOME/outputs:/app/outputs \
  rfdiffusion-api

# Test
curl http://localhost:8000/health
```

### Simple Design
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"inference.num_designs": 1, "contigmap.contigs": ["100-100"]}'
```

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/inference",
    json={"inference.num_designs": 1, "contigmap.contigs": ["50-50"]}
)

result = response.json()
print(f"Job {result['job_id']}: {result['status']}")
```

## Testing

### Automated Tests
```bash
# Quick test (no inference)
python scripts/test_api.py --skip-inference

# Full test with inference
python scripts/test_api.py --num-designs 1
```

### Example Script
```bash
# Run interactive examples
bash examples/api_usage.sh
```

## Backward Compatibility

The original `run_inference.py` script is **unchanged** and can still be used:

### CLI Mode (Original)
```bash
docker run --gpus all \
  --entrypoint python3.9 \
  rfdiffusion-api \
  scripts/run_inference.py \
  inference.num_designs=1 \
  'contigmap.contigs=[100-100]'
```

### Direct Script Execution
```bash
python scripts/run_inference.py \
  inference.num_designs=1 \
  'contigmap.contigs=[100-100]'
```

## Performance Considerations

1. **GPU Memory**: Each inference requires significant GPU memory
2. **Concurrent Requests**: Limited by GPU memory; recommend sequential processing
3. **Diffusion Steps**: Reduce `diffuser.T` for faster testing (default: 50)
4. **File Storage**: Trajectory files can be very large; use sparingly

## Security Considerations

1. **File Access**: Download endpoint restricted to `OUTPUT_DIR`
2. **Input Validation**: Basic validation on configuration parameters
3. **CORS**: Enabled by default; restrict in production if needed
4. **Network**: Container exposes port 8000; use firewall rules as needed

## Next Steps

1. **Read Documentation**: 
   - `QUICKSTART.md` for setup
   - `API_DOCUMENTATION.md` for API reference

2. **Test the API**:
   - Run `python scripts/test_api.py`
   - Try `bash examples/api_usage.sh`

3. **Deploy**:
   - Use `docker-compose up -d` for production
   - Configure reverse proxy (nginx) if needed
   - Set up monitoring and logging

4. **Integrate**:
   - Use Python requests library
   - Build web frontend
   - Create workflow automation

## Support

- **Original Project**: https://github.com/RosettaCommons/RFdiffusion
- **Issues**: Report in your deployment repository
- **Documentation**: See `API_DOCUMENTATION.md` and `QUICKSTART.md`

## License

Same as original RFdiffusion project.
