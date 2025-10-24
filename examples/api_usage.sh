#!/bin/bash
# Example API usage scripts for RFdiffusion REST API

# Configuration
API_URL="http://localhost:8000"

echo "=================================="
echo "RFdiffusion API Usage Examples"
echo "=================================="
echo ""

# 1. Health Check
echo "1. Health Check"
echo "Command:"
echo "curl $API_URL/health"
echo ""
echo "Response:"
curl -s $API_URL/health | jq '.' 2>/dev/null || curl -s $API_URL/health
echo ""
echo ""

# 2. Unconditional Design - 50 residues
echo "2. Unconditional Design (50 residues)"
echo "Command:"
cat << 'EOF'
curl -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 1,
    "contigmap.contigs": ["50-50"],
    "diffuser.T": 20
  }'
EOF
echo ""
echo "Submitting..."
RESPONSE=$(curl -s -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 1,
    "contigmap.contigs": ["50-50"],
    "diffuser.T": 20
  }')
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
echo ""
echo ""

# Extract job details
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id' 2>/dev/null)
PDB_FILE=$(echo "$RESPONSE" | jq -r '.results[0].pdb_file' 2>/dev/null | rev | cut -d'/' -f1 | rev)

if [ ! -z "$PDB_FILE" ] && [ "$PDB_FILE" != "null" ]; then
    # 3. List Designs
    echo "3. List Generated Designs"
    echo "Command:"
    echo "curl $API_URL/api/v1/designs"
    echo ""
    echo "Response:"
    curl -s $API_URL/api/v1/designs | jq '.designs | .[0:3]' 2>/dev/null || curl -s $API_URL/api/v1/designs
    echo ""
    echo ""

    # 4. Download PDB
    echo "4. Download PDB File"
    echo "Command:"
    echo "curl -O $API_URL/api/v1/download/$PDB_FILE"
    echo ""
    echo "Downloading..."
    curl -s -O $API_URL/api/v1/download/$PDB_FILE
    if [ -f "$PDB_FILE" ]; then
        echo "✓ Downloaded: $PDB_FILE"
        ls -lh "$PDB_FILE"
    else
        echo "✗ Download failed"
    fi
    echo ""
fi

echo "=================================="
echo "More Examples"
echo "=================================="
echo ""

# Example 3: Variable length design
echo "5. Variable Length Design (80-100 residues)"
cat << 'EOF'
curl -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 2,
    "contigmap.contigs": ["80-100"]
  }'
EOF
echo ""
echo ""

# Example 4: Multiple designs
echo "6. Generate Multiple Designs"
cat << 'EOF'
curl -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 5,
    "contigmap.contigs": ["100-100"],
    "inference.deterministic": true
  }'
EOF
echo ""
echo ""

# Example 5: Motif scaffolding (requires input PDB)
echo "7. Motif Scaffolding (requires input PDB)"
cat << 'EOF'
# First download a PDB file:
wget -P $HOME/inputs https://files.rcsb.org/view/5TPN.pdb

# Then run scaffolding:
curl -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 3,
    "inference.input_pdb": "/app/inputs/5TPN.pdb",
    "contigmap.contigs": ["10-40/A163-181/10-40"]
  }'
EOF
echo ""
echo ""

# Example 6: Fast testing with reduced steps
echo "8. Fast Testing (reduced diffusion steps)"
cat << 'EOF'
curl -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 1,
    "contigmap.contigs": ["30-30"],
    "diffuser.T": 10
  }'
EOF
echo ""
echo ""

# Example 7: With trajectory
echo "9. Generate with Trajectory (creates large files)"
cat << 'EOF'
curl -X POST $API_URL/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "inference.num_designs": 1,
    "contigmap.contigs": ["50-50"],
    "inference.write_trajectory": true
  }'
EOF
echo ""
echo ""

echo "=================================="
echo "Python Client Example"
echo "=================================="
cat << 'PYTHON'
import requests
import json

API_URL = "http://localhost:8000"

# Submit inference job
config = {
    "inference.num_designs": 2,
    "contigmap.contigs": ["60-60"],
    "diffuser.T": 25
}

print("Submitting job...")
response = requests.post(
    f"{API_URL}/api/v1/inference",
    json=config
)

result = response.json()
print(f"Job ID: {result['job_id']}")
print(f"Status: {result['status']}")

# Download results
for design in result['results']:
    filename = design['pdb_file'].split('/')[-1]
    print(f"Downloading {filename}...")
    
    pdb_response = requests.get(
        f"{API_URL}/api/v1/download/{filename}"
    )
    
    with open(filename, 'wb') as f:
        f.write(pdb_response.content)
    
    print(f"  Saved: {filename}")
    print(f"  Mean pLDDT: {design['mean_plddt']:.3f}")
    print(f"  Time: {design['time_seconds']:.1f}s")
PYTHON
echo ""
echo ""

echo "=================================="
echo "For full documentation, see:"
echo "  - API_DOCUMENTATION.md"
echo "  - QUICKSTART.md"
echo "=================================="
