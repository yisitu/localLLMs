Experience doc: https://docs.google.com/document/d/1P79r3jBOePrpSInZU4fK9BnQgK_o7Xg6lgde_SSxk6Q/edit?tab=t.0

```
sudo apt-get install python3.12-dev
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install accelerate
pip3 install wheel
pip3 install --no-build-isolation -v "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"

# Build causal-conv1d from source
#  git clone https://github.com/Dao-AILab/causal-conv1d.git
#  cd causal-conv1d
#  python3 setup.py build
#  python3 setup.py install

huggingface-cli login
```

```
Fri Nov 28 16:12:34 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.07             Driver Version: 581.80         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:01:00.0  On |                  N/A |
|  0%   35C    P8             35W /  575W |    2225MiB /  32607MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
