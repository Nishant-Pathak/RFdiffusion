# RFdiffusion REST API - Troubleshooting Guide

## Common Issues and Solutions

### 1. "RuntimeError: Numpy is not available"

**Error Message:**
```
RuntimeError: Numpy is not available
  at torch/_tensor.py in __array__
```

**Cause:**
This error occurs when PyTorch tensors are being passed where NumPy arrays or native Python types are expected. This can happen when Hydra/OmegaConf configuration values are not properly resolved to native Python types.

**Solution:**
This has been fixed in version with `convert_config_to_native_types()` function that ensures all configuration values are converted to native Python types before being passed to the inference pipeline.

**Verification:**
The API server now automatically handles this conversion. If you still encounter this issue:
1. Ensure you're using the latest version of `api_server.py`
2. Check that all numeric configuration parameters are passed as proper JSON numbers (not strings)
3. Restart the Docker container to ensure clean state

**Example - Correct JSON format:**
```json
{
  "inference.num_designs": 1,
  "diffuser.T": 50,
  "diffuser.min_sigma": 0.02,
  "diffuser.max_sigma": 1.5,
  "contigmap.contigs": ["100-100"]
}
```

**Example - Incorrect (don't do this):**
```json
{
  "inference.num_designs": "1",  // Wrong: string instead of number
  "diffuser.T": "50",             // Wrong: string instead of number
}
```

---

### 2. "No GPU detected" / Running on CPU

**Symptoms:**
- API health check shows `"gpu_available": false`
- Inference is extremely slow
- Logs show "Running on CPU"

**Solution:**
1. Verify nvidia-docker2 is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
   ```

2. Ensure you're using the `--gpus all` flag:
   ```bash
   docker run --gpus all ... rfdiffusion-api
   ```

3. Check GPU drivers and CUDA:
   ```bash
   nvidia-smi
   ```

---

### 3. "Model files not found"

**Error Message:**
```
FileNotFoundError: Model checkpoint not found
```

**Solution:**
1. Download models:
   ```bash
   bash scripts/download_models.sh $HOME/models
   ```

2. Verify models are mounted:
   ```bash
   docker exec rfdiffusion-api ls -la /app/models
   ```

3. Check volume mapping in docker run command:
   ```bash
   -v $HOME/models:/app/models
   ```

---

### 4. "Permission denied" writing outputs

**Symptoms:**
- Inference fails with permission errors
- Cannot write PDB files
- "Permission denied" in logs

**Solution:**
1. Fix permissions on output directory:
   ```bash
   chmod -R 777 $HOME/outputs
   ```

2. Or run container with user permissions:
   ```bash
   docker run --user $(id -u):$(id -g) ... rfdiffusion-api
   ```

---

### 5. "Port 8000 already in use"

**Error Message:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**Solution:**
1. Use a different port:
   ```bash
   docker run -p 9000:8000 ... rfdiffusion-api
   ```
   Then access at `http://localhost:9000`

2. Or stop the conflicting service:
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

---

### 6. Out of Memory (OOM) Errors

**Symptoms:**
- Inference crashes mid-way
- "CUDA out of memory" errors
- Container restarts unexpectedly

**Solution:**
1. Reduce protein size:
   ```json
   {"contigmap.contigs": ["50-50"]}  // Instead of 200+
   ```

2. Reduce diffusion steps:
   ```json
   {"diffuser.T": 20}  // Instead of default 50
   ```

3. Run one design at a time:
   ```json
   {"inference.num_designs": 1}
   ```

4. Ensure no other GPU processes are running:
   ```bash
   nvidia-smi
   ```

---

### 7. "Connection refused" or "Cannot connect to API"

**Symptoms:**
- `curl http://localhost:8000/health` fails
- Connection timeout errors

**Solution:**
1. Check container is running:
   ```bash
   docker ps
   ```

2. Check container logs:
   ```bash
   docker logs rfdiffusion-api
   ```

3. Verify port mapping:
   ```bash
   docker port rfdiffusion-api
   ```

4. Check firewall rules (if accessing remotely)

---

### 8. Slow inference / Performance issues

**Symptoms:**
- Each design takes very long (>10 minutes for small proteins)
- GPU utilization is low

**Possible Causes & Solutions:**

**A. Running on CPU instead of GPU:**
- Check GPU access with `--gpus all` flag
- Verify with health endpoint: `curl http://localhost:8000/health`

**B. Too many diffusion steps:**
- Reduce steps for testing: `{"diffuser.T": 10}`
- Default is 50 steps

**C. Large protein design:**
- Start with smaller proteins for testing
- Example: `{"contigmap.contigs": ["50-50"]}`

**D. Disk I/O bottleneck:**
- Ensure volumes are on fast storage (SSD preferred)
- Disable trajectory writing: `{"inference.write_trajectory": false}`

---

### 9. Hydra configuration errors

**Error Messages:**
```
omegaconf.errors.ConfigAttributeError: ...
```

**Solution:**
1. Check parameter names match exactly (case-sensitive):
   - Correct: `inference.num_designs`
   - Wrong: `inference.numDesigns`

2. Use proper nested notation:
   - Correct: `diffuser.T`
   - Wrong: `diffuser[T]`

3. Verify parameter exists in config:
   - Check `config/inference/base.yaml`

---

### 10. JSON parsing errors

**Error Message:**
```
400 Bad Request: No configuration provided
```

**Solution:**
1. Ensure proper JSON format:
   ```bash
   # Correct
   curl -X POST http://localhost:8000/api/v1/inference \
     -H "Content-Type: application/json" \
     -d '{"inference.num_designs": 1}'
   
   # Wrong - missing quotes
   curl -X POST http://localhost:8000/api/v1/inference \
     -H "Content-Type: application/json" \
     -d '{inference.num_designs: 1}'
   ```

2. Use a JSON validator for complex payloads

3. In Python, use `json.dumps()` or pass dict directly to `requests.post()`

---

## Debugging Tips

### Enable verbose logging
```bash
docker logs -f rfdiffusion-api
```

### Check container resources
```bash
docker stats rfdiffusion-api
```

### Inspect container
```bash
docker exec -it rfdiffusion-api /bin/bash
```

### Test with minimal configuration
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 1,
    "contigmap.contigs": ["30-30"],
    "diffuser.T": 5
  }'
```

### Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

---

## Getting Help

1. **Check logs first:**
   ```bash
   docker logs rfdiffusion-api --tail 100
   ```

2. **Test with provided examples:**
   ```bash
   python scripts/test_api.py --skip-inference
   bash examples/api_usage.sh
   ```

3. **Verify basic functionality:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Review documentation:**
   - `API_DOCUMENTATION.md` - API reference
   - `QUICKSTART.md` - Setup guide
   - `API_CHANGES.md` - Implementation details

5. **Check original RFdiffusion issues:**
   - https://github.com/RosettaCommons/RFdiffusion/issues

---

## Reporting Issues

When reporting issues, please include:

1. **Error message and full traceback**
2. **Docker version:** `docker --version`
3. **GPU info:** `nvidia-smi` output
4. **API request that caused the error**
5. **Relevant logs:** `docker logs rfdiffusion-api`
6. **System info:** OS, CUDA version, GPU model

Example:
```bash
# Collect debug info
echo "=== Docker Version ===" > debug.txt
docker --version >> debug.txt
echo "=== GPU Info ===" >> debug.txt
nvidia-smi >> debug.txt
echo "=== Container Logs ===" >> debug.txt
docker logs rfdiffusion-api --tail 200 >> debug.txt
```
