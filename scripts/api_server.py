#!/usr/bin/env python
"""
REST API server for RFdiffusion inference.

This server exposes RFdiffusion functionality via REST API endpoints.
"""

import base64
from datetime import datetime
import os
import re
import time
import pickle
import logging
import json
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
from google.cloud import pubsub_v1


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
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/results")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/inference"))

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_production_environment() -> bool:
	return os.getenv("ENVIRONMENT", "development") == "production"

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
    try:
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
    except Exception as e:
         log.debug(f"❌ Failed to update step run started_at for step run {step_run_id}: {e}")


def run_interface_wrapper(config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    job_id = config_overrides.get("job_id")
    step_run_id = config_overrides.get("step_run_id")
    update_step_run_started_at(step_run_id, get_database_engine())
    status = "SUCCESS"
    error_msg = None
    try:
        result = run_inference(config_overrides)
        log.debug(f"✅ Inference job {job_id} completed successfully with result: {result}")
        update_database_on_success(step_run_id, job_id)
        return result
    except Exception as e:
        log.error(f"Inference job {job_id} failed: {e}", exc_info=True)
        # Optionally, update the database to mark the step run as failed
        update_database_on_failure(step_run_id, e)
        status = "FAILED"
        error_msg = str(e)
        raise e
    finally:
        log.info(f"Preparing notification for job {job_id} with status {status}")
        notification_body = {
				"job_id": job_id,
				"status": status,
				"step_run_id": step_run_id,
			}
        if error_msg:
            notification_body["error_message"] = error_msg  
        
        if is_production_environment():
            if status == "FAILED" and should_skip_failure_notification(step_run_id):
                notification_body["give_up_retry"] = True
                return notification_body

            step_completed_publisher.publish(notification_body)
            log.info(f"Published completion notification for job {job_id} with status {status}")


def update_database_on_success(step_run_id: str, job_id: str):
    try:
        engine = get_database_engine()

        create_artifact_record(step_run_id, job_id, engine)
        insert_job_result_record(step_run_id, engine)
        update_step_run_completed(step_run_id, "SUCCESS", engine)
    except Exception as e:
        log.error(f"❌ Failed to update database on success for step run {step_run_id}: {e}")
        raise e


# insert into result set id=uuid_generate_v4(), step_run_id=step_run_id, created_at=now();
def insert_job_result_record(step_run_id: str, engine):
	# Create fresh metadata to avoid caching issues
	metadata = MetaData()
	job_results_table = Table('result', metadata, autoload_with=engine)

	with engine.connect() as connection:

		# check if result already exists for this step_run_id
		select_stmt = (
			select(job_results_table.c.id).
			where(job_results_table.c.step_run_id == step_run_id)
		)
		result = connection.execute(select_stmt).fetchone()
		stmt = None
		if result:
			log.info(f"ℹ️  Job result record already exists for step run id {step_run_id}, skipping insertion.")
			stmt = update(job_results_table).where(job_results_table.c.step_run_id == step_run_id).values(
				created_at=datetime.utcnow()
			)
		else:
			stmt = insert(job_results_table).values(
				id=str(uuid.uuid4()),
				step_run_id=step_run_id,
				created_at=datetime.utcnow()
		)

		connection.execute(stmt)
		connection.commit()
		log.info(f"✅ Created job result record for step run id {step_run_id}")

def create_artifact_record(step_run_id: str, job_id: str, engine):
	"""
	Create artifact records for all design files in the job output directory.
	Finds all files starting with 'design_' including .pdb, .trb, and trajectory files.
	
	Args:
		step_run_id: The step run ID
		job_id: The job ID used to construct the output directory path
		engine: SQLAlchemy database engine
	"""
	# Create fresh metadata to avoid caching issues
	metadata = MetaData()
	artifacts_table = Table('artifact', metadata, autoload_with=engine)
	
	# Construct the job output directory path
	job_output_dir = os.path.join(OUTPUT_DIR, job_id)
	
	if not os.path.exists(job_output_dir):
		log.warning(f"⚠️  Job output directory does not exist: {job_output_dir}")
		return
	
	# Find all files starting with 'design_'
	design_files = []
	
	# Find design files in the main directory (design_*.pdb, design_*.trb)
	for pattern in ['design_*.pdb', 'design_*.trb']:
		design_files.extend(glob.glob(os.path.join(job_output_dir, pattern)))
	
	# Find trajectory files in the traj subdirectory
	traj_dir = os.path.join(job_output_dir, 'traj')
	if os.path.exists(traj_dir):
		for pattern in ['design_*_Xt-1_traj.pdb', 'design_*_pX0_traj.pdb']:
			design_files.extend(glob.glob(os.path.join(traj_dir, pattern)))
	
	if not design_files:
		log.warning(f"⚠️  No design files found in {job_output_dir}")
		return
	
	log.info(f"Found {len(design_files)} design files to create artifacts for")
	
	with engine.connect() as connection:
		for file_path in design_files:
			try:
				filename = os.path.basename(file_path)
				file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
				
				# Determine mime type and artifact type based on file extension
				if file_path.endswith('.pdb'):
					mime_type = 'chemical/x-pdb'
					if 'traj' in file_path:
						artifact_type = 'TRAJECTORY_PDB'
					else:
						artifact_type = 'DESIGN_PDB'
				elif file_path.endswith('.trb'):
					mime_type = 'application/octet-stream'
					artifact_type = 'DESIGN_TRB'
				else:
					mime_type = 'application/octet-stream'
					artifact_type = 'RAW_OUTPUT'
				
				# Check if artifact already exists for this step_run_id and filename
				select_stmt = (
					select(artifacts_table.c.id).
					where(artifacts_table.c.step_run_id == step_run_id).
					where(artifacts_table.c.filename == filename)
				)
				result = connection.execute(select_stmt).fetchone()
				
				stmt = None
				if result:
					log.info(f"ℹ️  Artifact record already exists for {filename}, updating instead.")
					stmt = (
						update(artifacts_table).
						where(artifacts_table.c.step_run_id == step_run_id).
						where(artifacts_table.c.filename == filename).
						values(
							file_path=file_path,
							file_size=file_size,
							mime_type=mime_type,
							artifact_type=artifact_type,
							storage_backend='GCS' if is_production_environment() else 'LOCAL',
							is_public=False,
							created_at=datetime.utcnow()
						)
					)
					log.info(f"✅ Updated artifact record for {filename}")
				else:
					stmt = insert(artifacts_table).values(
						id=uuid.uuid4(),
						step_run_id=step_run_id,
						filename=filename,
						file_path=file_path,
						file_size=file_size,
						mime_type=mime_type,
						artifact_type=artifact_type,
						storage_backend='GCS' if is_production_environment() else 'LOCAL',
						is_public=False,
						created_at=datetime.utcnow()
					)
					log.info(f"✅ Created artifact record for {filename}")
				
				connection.execute(stmt)
			except Exception as e:
				log.error(f"❌ Failed to create artifact record for {file_path}: {e}")
				continue
		
		connection.commit()
		log.info(f"✅ Successfully processed {len(design_files)} artifact records for step run {step_run_id}")


def update_database_on_failure(step_run_id: str, error: Exception):
    try:
        engine = get_database_engine()

        update_step_run_completed(step_run_id, "FAILED", engine, str(error))
    except Exception as e:
        log.debug(f"❌ Failed to update database on failure for step run {step_run_id}: {e}")


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

def should_skip_failure_notification(step_run_id: str) -> bool:
	"""
	Check if failure notification should be skipped based on retry count.
	Returns True if retries are remaining, False if notification should be sent.
	"""
	try:
		engine = get_database_engine()
		metadata = MetaData()
		steprun_table = Table('steprun', metadata, autoload_with=engine)

		with engine.connect() as connection:
			select_stmt = (
				select(steprun_table.c.retry_count, steprun_table.c.max_retries).
				where(steprun_table.c.id == step_run_id)
			)
			result = connection.execute(select_stmt).fetchone()

			if result:
				retry_count, max_retries = result
				if retry_count < max_retries:
					log.info(f"Skipping failure notification for step_run_id {step_run_id} as retries are remaining ({retry_count}/{max_retries})")
					return True

			return False
	except Exception as e:
		log.error(f"❌ Failed to check retry count: {e}")
		# On error, don't skip notification (fail safe)
		return False


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
    pdb_file_path = config_overrides.get("pdb_file_path")
    config_overrides.pop("job_id", None)
    config_overrides.pop("step_run_id", None)
    config_overrides.pop("pdb_file_path", None)
    config_overrides['inference.input_pdb'] = pdb_file_path

    # Initialize Hydra
    initialize_hydra_config(config_overrides.get("config_name", "base"))
    
    # Prepare overrides list
    overrides = []
    for key, value in config_overrides.items():
        if key != "config_name":
            overrides.append(f"{key}={value}")
    
    # Compose configuration
    conf = compose(config_name=config_overrides.get("config_name", "base"), overrides=overrides)
    
    # Create job-specific output directory with absolute path
    job_output_dir = os.path.abspath(os.path.join(OUTPUT_DIR, job_id))
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Set output prefix if not provided - always use absolute path
    if "inference.output_prefix" not in config_overrides:
        conf.inference.output_prefix = os.path.join(job_output_dir, "design")
    else:
        # Ensure custom output prefix is also within job directory
        conf.inference.output_prefix = os.path.join(job_output_dir, os.path.basename(config_overrides["inference.output_prefix"]))
    
    # Set model directory if not provided
    if "inference.model_directory_path" not in config_overrides:
        conf.inference.model_directory_path = MODEL_DIR
    
    log.info(f"Starting inference job {job_id}")
    log.info(f"Job output directory: {job_output_dir}")
    log.info(f"Output prefix: {conf.inference.output_prefix}")
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
        
        # Save outputs - all files go into job_output_dir/{job_id}
        final_seq = seq_stack[-1]
        
        # Set glycines for non-motif regions
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )
        
        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        
        # Ensure output directory exists
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Write PDB file - use relative path within job directory
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
            traj_dir = os.path.join(job_output_dir, "traj")
            os.makedirs(traj_dir, exist_ok=True)
            traj_prefix = os.path.join(traj_dir, os.path.basename(out_prefix))
            
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
        message_data = base64.b64decode(pubsub_message['data']).decode('utf-8')
        data = json.loads(message_data)
        log.info(f"Received Pub/Sub message: {data}")
        
        job_id = data.get("job_id")
        step_run_id = data.get("step_run_id")

        # Process the message
        # Run inference
        interface_input = {
            "config_name": data.get("config_name", "base"),
            "inference.num_designs": data.get("num_designs", 1),
        }
        
        # Make contigs configurable from the message
        # If not provided, use None to let RFdiffusion use default behavior
        if "contigs" in data:
            interface_input["contigmap.contigs"] = data["contigs"]
        
        # Add other optional parameters from the message
        if "num_designs" in data:
            interface_input["inference.num_designs"] = data["num_designs"]
        
        if "write_trajectory" in data:
            interface_input["inference.write_trajectory"] = data["write_trajectory"]
        
        interface_input["pdb_file_path"] = data.get("pdb_file_path")
        interface_input["job_id"] = job_id
        interface_input["step_run_id"] = step_run_id
        result = run_interface_wrapper(interface_input)

        result["job_id"] = job_id
        result["step_run_id"] = step_run_id  

                # Check if processing was successful
        if result.get('status') == 'SUCCESS':
            # Return 200 to ACK the message
            log.info(f"Successfully processed message for job {result.get('job_id')}")
            return jsonify(result), 200
        elif 'give_up_retry' in result and result['give_up_retry']:
            log.debug("All the retry limit exausted")
            return jsonify({'status': 'FAILED', 'reason': 'Retry Exausted'}), 200
        else:
            # Return 500 to NACK the message (will be retried)
            log.error(f"Failed to process message for job {result.get('job_id')}: {result.get('error_message')}")
            return jsonify(result), 500
      
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
        result = run_interface_wrapper(config_overrides)

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



class Publisher:
	def __init__(self, topic_name: str):
		self.topic = topic_name
		self.publisher = pubsub_v1.PublisherClient()

	def publish(self, message: dict, **attrs):
		if not self.topic:
			raise ValueError("PUBSUB_TOPIC environment variable not set.")
		import json
		message = json.dumps(message)
		data = message.encode("utf-8")
		future = self.publisher.publish(self.topic, data, **{str(k): str(v) for k, v in attrs.items()})
		return future.result()

STEP_COMPLETED_TOPIC = 'projects/alphafold-469417/topics/step-completed'

step_completed_publisher = Publisher(STEP_COMPLETED_TOPIC)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    
    log.info(f"Starting RFdiffusion API server on {host}:{port}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(f"Model directory: {MODEL_DIR}")
    
    app.run(host=host, port=port, debug=False)
