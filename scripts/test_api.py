#!/usr/bin/env python3
"""
Test script for RFdiffusion REST API.

Usage:
    python test_api.py --host localhost --port 8000
"""

import argparse
import requests
import json
import time
import sys


def test_health(base_url):
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed")
        print(f"  Status: {data['status']}")
        print(f"  GPU Available: {data['gpu_available']}")
        print(f"  GPU Name: {data['gpu_name']}")
        print(f"  Output Dir: {data['output_dir']}")
        print(f"  Model Dir: {data['model_dir']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_inference(base_url, num_designs=1):
    """Test the inference endpoint."""
    print(f"\nTesting /api/v1/inference endpoint ({num_designs} design(s))...")
    
    config = {
        "inference.num_designs": num_designs,
        "contigmap.contigs": ["50-50"],  # Simple unconditional design
        "inference.deterministic": True,
        "diffuser.T": 10,  # Reduced steps for faster testing
    }
    
    try:
        print("Submitting inference job...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/v1/inference",
            json=config,
            headers={"Content-Type": "application/json"},
            timeout=3600  # 1 hour timeout
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        data = response.json()
        
        print(f"✓ Inference completed in {elapsed:.2f} seconds")
        print(f"  Job ID: {data['job_id']}")
        print(f"  Status: {data['status']}")
        print(f"  Num Designs: {data['num_designs']}")
        
        for result in data['results']:
            print(f"\n  Design {result['design_number']}:")
            print(f"    PDB: {result['pdb_file']}")
            print(f"    Time: {result['time_seconds']:.2f}s")
            print(f"    Mean pLDDT: {result['mean_plddt']:.3f}")
        
        return True, data
        
    except requests.exceptions.Timeout:
        print(f"✗ Inference timed out")
        return False, None
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        try:
            print(f"  Response: {response.json()}")
        except:
            pass
        return False, None


def test_list_designs(base_url):
    """Test the list designs endpoint."""
    print("\nTesting /api/v1/designs endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/designs", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ List designs passed")
        print(f"  Total designs: {data['total']}")
        
        if data['total'] > 0:
            print(f"\n  Recent designs:")
            for design in data['designs'][:5]:  # Show first 5
                print(f"    - {design['pdb_file']} ({design['size_bytes']} bytes)")
        
        return True
    except Exception as e:
        print(f"✗ List designs failed: {e}")
        return False


def test_download(base_url, filename):
    """Test the download endpoint."""
    print(f"\nTesting /api/v1/download endpoint...")
    try:
        response = requests.get(
            f"{base_url}/api/v1/download/{filename}",
            timeout=60
        )
        response.raise_for_status()
        
        print(f"✓ Download passed")
        print(f"  File: {filename}")
        print(f"  Size: {len(response.content)} bytes")
        
        # Optionally save the file
        output_file = f"test_downloaded_{filename}"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"  Saved to: {output_file}")
        
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test RFdiffusion REST API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--num-designs", type=int, default=1, help="Number of designs for inference test")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the inference test (takes time)")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download test")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print(f"Testing RFdiffusion API at {base_url}")
    print("=" * 60)
    
    results = {}
    
    # Test health endpoint
    results['health'] = test_health(base_url)
    
    if not results['health']:
        print("\n" + "=" * 60)
        print("Health check failed. API may not be running.")
        print("=" * 60)
        sys.exit(1)
    
    # Test list designs endpoint
    results['list'] = test_list_designs(base_url)
    
    # Test inference endpoint
    inference_data = None
    if not args.skip_inference:
        results['inference'], inference_data = test_inference(base_url, args.num_designs)
    else:
        print("\nSkipping inference test (--skip-inference)")
    
    # Test download endpoint
    if not args.skip_download and inference_data and results.get('inference'):
        # Try to download the first result
        if inference_data['results']:
            pdb_path = inference_data['results'][0]['pdb_file']
            filename = pdb_path.split('/')[-1]
            results['download'] = test_download(base_url, filename)
    elif not args.skip_download:
        print("\nSkipping download test (no inference data available)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
