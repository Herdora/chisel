#!/bin/bash
set -e

###############################################################################
# system prep
###############################################################################
apt-get update -y
apt-get install -y python3-venv build-essential cmake git python3-pip \
                   wget curl gnupg software-properties-common

mkdir -p /workspace && chmod 755 /workspace

###############################################################################
# TODO: add rocm-specific profilers / other tools (curr digital ocean image should have them)
###############################################################################


###############################################################################
# python virtual-env
###############################################################################
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip setuptools wheel

###############################################################################
# upload requirements.txt to workspace
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
networkx==3.5
numpy==2.3.0
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
# install requirements
###############################################################################
pip install -r /workspace/requirements.txt

###############################################################################
# install rocm-specific profilers / other tools
###############################################################################
mkdir -p /workspace/rocm-wheels && cd /workspace/rocm-wheels
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchvision-0.21.0%2Brocm6.4.1.git4040d51f-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchaudio-2.6.0%2Brocm6.4.1.gitd8831425-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl
pip install --no-cache-dir torch-*.whl torchvision-*.whl torchaudio-*.whl pytorch_triton_rocm-*.whl
cd /workspace && rm -rf rocm-wheels

###############################################################################
# environment convenience
###############################################################################
echo 'export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/' >> /root/.bashrc
echo 'source /opt/venv/bin/activate'              >> /root/.bashrc
echo 'cd /workspace'                              >> /root/.bashrc
echo 'source ~/.bashrc'                           >> /root/.bash_profile

echo "✓ Cloud-init finished"
