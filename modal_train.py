import subprocess
import modal

# Create the Modal application
app = modal.App("bayesian-cnn")

# Persistent storage for trained checkpoints. Without this, anything written
# inside the container's filesystem (including checkpoints/) is destroyed the
# moment the function returns — that's why your last run "vanished".
volume = modal.Volume.from_name("bayesiancnn-checkpoints", create_if_missing=True)

CHECKPOINT_DIR = "/root/project/checkpoints"

# Build the execution environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "Pillow",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            ".git",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".idea",
            ".vscode",
            ".github",
            "checkpoints",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
        ],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 12,  # 12 hours
    volumes={CHECKPOINT_DIR: volume},
)
def train(dataset: str = "CIFAR10", net_type: str = "3conv3fc"):

    # Confirms the container actually has a GPU attached before you wait
    # hours for a run — if this fails/prints nothing useful, the job is
    # running on CPU regardless of gpu="A10G" being set.
    subprocess.run(["nvidia-smi"], cwd="/root/project")

    print(f"Starting Bayesian CNN training: dataset={dataset}, net_type={net_type}")

    # Popen + streaming instead of subprocess.run(capture_output=True):
    # capture_output buffers everything and only prints it after the
    # process exits, so a multi-hour job would show nothing in the Modal
    # dashboard until it was already finished (or silently hung).
    process = subprocess.Popen(
        ["python", "main_bayesian.py", "--dataset", dataset, "--net_type", net_type],
        cwd="/root/project",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed with exit code {process.returncode}"
        )

    # Persist everything written to CHECKPOINT_DIR during this run back to
    # the Modal Volume so it survives after the container is torn down.
    volume.commit()
    print(f"Checkpoints committed to volume 'bayesiancnn-checkpoints'.")


@app.local_entrypoint()
def main(dataset: str = "CIFAR10", net_type: str = "3conv3fc"):
    train.remote(dataset=dataset, net_type=net_type)