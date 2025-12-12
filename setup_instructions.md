# TPU VM Setup Instructions

## 1. SSH into the TPU VM

```bash
gcloud alpha compute tpus tpu-vm ssh v6e-8-node-15 --zone us-east1-d
```

## 2. Stop any existing processes using port 8000

Check for running docker containers:
```bash
sudo docker ps
```

Stop the vllm-tpu container if running:
```bash
sudo docker stop vllm-tpu
```


## 3. Clone the repository

```bash
git clone https://github.com/OhadRubin/SkyRL.git
cd ~/SkyRL
```

## 4. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

Add to your `.bashrc` for persistence:
```bash
echo 'source ~/.local/bin/env' >> ~/.bashrc
```

## 5. Create virtual environment and install dependencies

```bash
cd ~/SkyRL/skyrl-tx
uv venv
uv sync --extra tpu --extra tinker
```

## 6. Run the Tinker server

```bash
cd ~/SkyRL
bash ~/SkyRL/run_tinker_server.sh --scan-layers
```

## Quick one-liner setup (after SSH)

```bash
git clone https://github.com/OhadRubin/SkyRL.git && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
source ~/.local/bin/env && \
cd ~/SkyRL/skyrl-tx && \
uv venv && \
uv sync --extra tpu --extra tinker && \
cd ~/SkyRL && \
bash ~/SkyRL/run_tinker_server.sh --scan-layers
```

## Updating and re-running

After initial setup, to pull changes and run:

```bash
source ~/.local/bin/env
cd ~/SkyRL
git pull
bash ~/SkyRL/run_tinker_server.sh --scan-layers
```

## Server flags

- `--scan-layers` - Use scan over layers to reduce memory (recommended for large models)
- `--use-ring-attention` - Enable ring attention for sequence parallelism
- `--scan-query-chunk-size N` - Query chunk size for ring attention (default: 512)
- `--scan-key-chunk-size N` - Key chunk size for ring attention (default: 512)
