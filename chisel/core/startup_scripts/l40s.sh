#!/bin/bash
set -e

###############################################################################
# System prep
###############################################################################
apt-get update -y
apt-get install -y python3-venv build-essential cmake git python3-pip \
                   wget curl gnupg software-properties-common

mkdir -p /workspace && chmod 755 /workspace

###############################################################################
# NVIDIA driver & CUDA toolkit 12.6
###############################################################################
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -y
apt-get install -y cuda-toolkit-12-6 nsight-compute nsight-systems || true

###############################################################################
# Optional NVIDIA profilers / comms libs (uncomment if you want them)
###############################################################################



###############################################################################
# Python virtual-env
###############################################################################
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip setuptools wheel

###############################################################################
# requirements.txt
###############################################################################
cat <<'EOF' >/workspace/requirements.txt
ansi2html==1.9.2
astunparse==1.6.2
azure-core==1.34.0
azure-identity==1.23.0
bcrypt==4.3.0
blinker==1.9.0
certifi==2025.6.15
cffi==1.17.1
charset-normalizer==3.4.2
chisel-cli==0.1.2
click==8.2.1
colorlover==0.3.0
contourpy==1.3.2
cryptography==45.0.4
cycler==0.12.1
dash==2.14.2
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-svg==0.0.12
dash-table==5.0.0
dnspython==2.7.0
filelock==3.18.0
Flask==3.0.3
fonttools==4.58.4
fsspec==2025.5.1
idna==3.10
importlib_metadata==8.7.0
iniconfig==2.1.0
isodate==0.7.2
itsdangerous==2.2.0
Jinja2==3.1.6
kaleido==0.2.1
kiwisolver==1.4.8
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.3
mdurl==0.1.2
mpmath==1.3.0
msal==1.32.3
msal-extensions==1.3.1
msrest==0.7.1
narwhals==1.43.1
nest-asyncio==1.6.0
networkx==3.4.2
numpy==2.2.6
oauthlib==3.3.1
packaging==25.0
pandas==2.3.0
paramiko==3.5.1
pillow==11.2.1
plotille==5.0.0
plotly==6.1.2
pluggy==1.6.0
py-cpuinfo==9.0.0
pycparser==2.22
pydo==0.11.0
Pygments==2.19.1
PyJWT==2.10.1
pymongo==4.13.2
PyNaCl==1.5.0
pyparsing==3.2.3
pytest==8.4.1
pytest-benchmark==5.1.0
python-dateutil==2.9.0.post0
tqdm==4.67.1
typer==0.16.0
typing_extensions==4.14.0
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.0.6
wheel==0.45.1
zipp==3.23.0
EOF

###############################################################################
# PyTorch CUDA 12.6 wheels
###############################################################################
echo "⇢ Installing PyTorch 2.7.x CUDA 12.6 wheels"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

###############################################################################
# Install requirements
###############################################################################
pip install -r /workspace/requirements.txt

###############################################################################
# environment convenience
###############################################################################
echo 'export CUDA_HOME=/usr/local/cuda'           >> /root/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin'      >> /root/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> /root/.bashrc
echo 'source /opt/venv/bin/activate'              >> /root/.bashrc
echo 'cd /workspace'                              >> /root/.bashrc
echo 'source ~/.bashrc'                           >> /root/.bash_profile

echo "✓ Cloud-init finished — PyTorch CUDA 12.6 environment ready"
