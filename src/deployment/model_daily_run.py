import modal
from pathlib import Path
import os
import yaml
import subprocess

app = modal.App("daily-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.12.11")
    .pip_install_from_requirements(requirements_txt="pyproject.toml")
    .add_local_dir("src/deployment", remote_path="/root/general")
)

if Path(".env").exists():
    from dotenv import dotenv_values
    env_vars = dotenv_values(".env")
else:
    env_vars = {}

@app.function(
    image=image,
    schedule=modal.Period(days=1),
    secrets=[modal.Secret.from_dict(env_vars)],
    timeout=1000,
)
def run_pipeline():

    print("--- Running Training ---")
    subprocess.run(["python", "/root/general/daily_training.py"], check=True)

    # Run Inference
    print("--- Running Inference ---")
    subprocess.run(["python", "/root/general/daily_inference.py"], check=True)