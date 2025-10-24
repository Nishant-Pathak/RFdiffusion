#!/usr/bin/env python
"""
REST API server for RFdiffusion inference.

This server exposes RFdiffusion functionality via REST API endpoints.
"""

from datetime import datetime
import os
import re
import time
import pickle
import logging
import tempfile
import traceback
import uuid
import glob
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import random
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from sqlalchemy import MetaData, Table, update, select, insert
import uuid
import shutil
from sqlalchemy.exc import SQLAlchemyError
from config import is_production_environment


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Global configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/inference"))

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_deterministic(seed=0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_hydra_config(config_name: str = "base") -> None:
    """Initialize Hydra configuration."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize_config_dir(config_dir=CONFIG_DIR, version_base=None)


def get_database_engine():
	from sqlalchemy import create_engine
	import os

	database_url = os.getenv("DATABASE_URL", "postgresql://postgres:tomatoA10@localhost:5432/protein_app")
	if not database_url:
		log.warning("⚠️  DATABASE_URL environment variable not set. Cannot create database engine.")
		raise ValueError("DATABASE_URL environment variable not set")
	
	try:
		engine = create_engine(database_url)
		return engine
	except Exception as e:
		log.error(f"❌ Failed to create database engine: {e}")
		log.error(f"Traceback: {traceback.format_exc()}")
		raise e


def update_step_run_started_at(step_run_id: str, engine):
	metadata = MetaData()
	steprun_table = Table('steprun', metadata, autoload_with=engine)
	
	with engine.connect() as connection:
		stmt = (
			update(steprun_table).
			where(steprun_table.c.id == step_run_id).
			values(
				status='RUNNING',
				started_at=datetime.utcnow()
			)
		)
		connection.execute(stmt)
		connection.commit()
		log.info(f"Updated step run started_at for step run {step_run_id}")


def run_interence_wrapper(config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    job_id = config_overrides.get("job_id")
    step_run_id = config_overrides.get("step_run_id")
    update_step_run_started_at(step_run_id, get_database_engine())
    try:
        run_inference(config_overrides)
        update_database_on_success(step_run_id)

    except Exception as e:
        log.error(f"Inference job {job_id} failed: {e}", exc_info=True)
        # Optionally, update the database to mark the step run as failed
        update_database_on_failure(step_run_id, e)
        raise e


def update_database_on_success(step_run_id: str, destination_path: str):
	engine = get_database_engine()

	# create_artifact_record(step_run_id, destination_path, engine)
	# insert_job_result_record(step_run_id, engine)
	update_step_run_completed(step_run_id, "SUCCESS", engine)



def update_database_on_failure(step_run_id: str, error: Exception):
	engine = get_database_engine()

	update_step_run_completed(step_run_id, "FAILED", engine, str(error))


def update_step_run_completed(step_run_id: str, status: str, engine, error_message: str = None):
	metadata = MetaData()
	steprun_table = Table('steprun', metadata, autoload_with=engine)
	
	with engine.connect() as connection:
		# First, fetch the started_at timestamp
		select_stmt = (
			select(steprun_table.c.started_at).
			where(steprun_table.c.id == step_run_id)
		)
		result = connection.execute(select_stmt).fetchone()
		
		execution_time = None
		if result and result[0]:
			started_at = result[0]
			execution_time = (datetime.utcnow() - started_at).total_seconds()
		
		stmt = (
			update(steprun_table).
			where(steprun_table.c.id == step_run_id).
			where(steprun_table.c.step_name == 'ALPHAFOLD').
			values(
				status=status,
				completed_at=datetime.utcnow(),
				execution_time=execution_time,
				error_message=error_message,
				retry_count=steprun_table.c.retry_count + 1 if status == "FAILED" else steprun_table.c.retry_count
			)
		)
		connection.execute(stmt)
		connection.commit()
		log.info(f"Updated step run status to {status} for step run id {step_run_id}")


def run_inference(config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run RFdiffusion inference with given configuration.
    
    Args:
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        Dictionary containing results and metadata
    """
    # Generate unique job ID
    job_id = config_overrides.get("job_id")
    step_run_id = config_overrides.get("step_run_id")

    # Initialize Hydra
    initialize_hydra_config(config_overrides.get("config_name", "base"))
    
    # Prepare overrides list
    overrides = []
    for key, value in config_overrides.items():
        if key != "config_name":
            overrides.append(f"{key}={value}")
    
    # Compose configuration
    conf = compose(config_name=config_overrides.get("config_name", "base"), overrides=overrides)
    
    # Set output prefix if not provided
    if "inference.output_prefix" not in config_overrides:
        conf.inference.output_prefix = os.path.join(OUTPUT_DIR, f"design_{job_id}")
    
    # Set model directory if not provided
    if "inference.model_directory_path" not in config_overrides:
        conf.inference.model_directory_path = MODEL_DIR
    
    log.info(f"Starting inference job {job_id}")
    log.info(f"Configuration: {OmegaConf.to_yaml(conf)}")
    
    # Check for deterministic mode
    if conf.inference.get("deterministic", False):
        make_deterministic()
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Using GPU: {device_name}")
    else:
        log.info("Running on CPU (no GPU detected)")
    
    # Initialize sampler
    sampler = iu.sampler_selector(conf)
    
    # Determine starting design number
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            m = re.match(".*_(\d+)\.pdb$", e)
            if m:
                indices.append(int(m.groups()[0]))
        design_startnum = max(indices) + 1
    
    results = []
    
    # Loop over number of designs
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.get("deterministic", False):
            make_deterministic(i_des)
        
        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        
        # Skip if file exists in cautious mode
        if sampler.inf_conf.get("cautious", False) and os.path.exists(out_prefix + ".pdb"):
            log.info(f"Skipping {out_prefix}.pdb (already exists)")
            continue
        
        # Sample initial state
        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []
        
        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        
        # Reverse diffusion loop
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])
        
        # Prepare outputs
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0])
        plddt_stack = torch.stack(plddt_stack)
        
        result_dir_path = Path(f"{OUTPUT_DIR}/{job_id}")
        os.makedirs(result_dir_path, exist_ok=True)

        # Save outputs
        os.makedirs(Path(f"{result_dir_path}/{out_prefix}"), exist_ok=True)
        final_seq = seq_stack[-1]
        
        # Set glycines for non-motif regions
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )
        
        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        
        # Write PDB file
        out_pdb = f"{out_prefix}.pdb"
        writepdb(
            out_pdb,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
            idx_pdb=sampler.idx_pdb
        )
        
        # Save metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)
        
        # Write trajectory if requested
        if sampler.inf_conf.get("write_trajectory", False):
            traj_prefix = os.path.dirname(result_dir_path) + "/" + os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
            
            writepdb_multi(
                f"{traj_prefix}_Xt-1_traj.pdb",
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
            
            writepdb_multi(
                f"{traj_prefix}_pX0_traj.pdb",
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
        
        elapsed_time = time.time() - start_time
        log.info(f"Finished design in {elapsed_time/60:.2f} minutes")
        
        results.append({
            "design_number": i_des,
            "pdb_file": out_pdb,
            "trb_file": f"{out_prefix}.trb",
            "time_seconds": elapsed_time,
            "mean_plddt": float(plddt_stack.mean().cpu().numpy())
        })
    
    # Clear Hydra instance
    GlobalHydra.instance().clear()
    
    return {
        "job_id": job_id,
        "status": "completed",
        "num_designs": len(results),
        "results": results
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    
    return jsonify({
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "output_dir": OUTPUT_DIR,
        "model_dir": MODEL_DIR
    })

@app.route('/predict', methods=['POST'])
def handle_pubsub_push():
    try:
        # Get the Pub/Sub message from the request
        envelope = request.get_json()
        
        if not envelope:
            log.error('No Pub/Sub message received')
            return 'Bad Request: no Pub/Sub message received', 400
        
        pubsub_message = envelope.get('message')
        log.info(f"Pub/Sub message: {pubsub_message}")
        if not pubsub_message:
            log.error('No message field in Pub/Sub envelope')
            return 'Bad Request: invalid Pub/Sub message format', 400
        
        # Decode the message data
        message_data = log.b64decode(pubsub_message['data']).decode('utf-8')
        data = log.loads(message_data)
        log.info(f"Received Pub/Sub message: {data}")
        
        job_id = data.get("job_id")
        step_run_id = data.get("step_run_id")

        # Process the message
        # Run inference
        result = run_interence_wrapper(data)

        result["job_id"] = job_id
        result["step_run_id"] = step_run_id
        
        return jsonify(result), 200
    except Exception as e:
        log.error(f"Error processing Pub/Sub message: {e}")
        # Return 500 to NACK the message (will be retried)
        return jsonify({
            'error': str(e),
            'status': 'FAILED'
        }), 500


@app.route('/rfdiffusion', methods=['POST'])
def inference():
    """
    Run RFdiffusion inference.
    
    Expected JSON payload:
    {
        "config_name": "base",  # optional, default: "base"
        "inference.num_designs": 1,
        "inference.input_pdb": "/path/to/input.pdb",  # optional
        "contigmap.contigs": ["10-40/A163-181/10-40"],  # optional
        ... other configuration parameters
    }
    
    Returns:
    {
        "job_id": "abc123",
        "status": "completed",
        "num_designs": 1,
        "results": [
            {
                "design_number": 0,
                "pdb_file": "/path/to/output.pdb",
                "trb_file": "/path/to/output.trb",
                "time_seconds": 123.45,
                "mean_plddt": 0.85
            }
        ]
    }
    """
    try:
        config_overrides = request.json
        if not config_overrides:
            return jsonify({"error": "No configuration provided"}), 400
        log.info(f"Received inference request: {config_overrides}")
        
        job_id = config_overrides.get("job_id")
        step_run_id = config_overrides.get("step_run_id")


        
        # Run inference
        result = run_interence_wrapper(config_overrides)

        result["job_id"] = job_id
        result["step_run_id"] = step_run_id
        
        return jsonify(result), 200
        
    except Exception as e:
        log.error(f"Inference failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


@app.route('/api/v1/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """
    Download generated files.
    
    Args:
        filename: Relative path to file in OUTPUT_DIR
    """
    try:
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Security check: ensure file is within OUTPUT_DIR
        if not os.path.abspath(file_path).startswith(os.path.abspath(OUTPUT_DIR)):
            return jsonify({"error": "Access denied"}), 403
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        log.error(f"Download failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/designs', methods=['GET'])
def list_designs():
    """List all generated designs in OUTPUT_DIR."""
    try:
        import glob
        pdb_files = glob.glob(os.path.join(OUTPUT_DIR, "**/*.pdb"), recursive=True)
        
        designs = []
        for pdb_file in pdb_files:
            rel_path = os.path.relpath(pdb_file, OUTPUT_DIR)
            trb_file = pdb_file.replace(".pdb", ".trb")
            
            designs.append({
                "pdb_file": rel_path,
                "has_trb": os.path.exists(trb_file),
                "size_bytes": os.path.getsize(pdb_file),
                "modified_time": os.path.getmtime(pdb_file)
            })
        
        return jsonify({
            "total": len(designs),
            "designs": designs
        }), 200
        
    except Exception as e:
        log.error(f"Failed to list designs: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    log.info(f"Starting RFdiffusion API server on {host}:{port}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(f"Model directory: {MODEL_DIR}")
    
    app.run(host=host, port=port, debug=False)
