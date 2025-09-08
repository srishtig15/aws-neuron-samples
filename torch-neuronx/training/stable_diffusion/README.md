```
source /opt/aws_neuronx_venv_pytorch_2_7_nxd_training/bin/activate

pip install -U -r requirements.txt

cp modeling_utils.py /opt/aws_neuronx_venv_pytorch_2_7_nxd_training/lib/python3.10/site-packages/diffusers/models/modeling_utils.py

XLA_PARAMETER_WRAPPING_THREADSHOLD=6400 python run.py
```