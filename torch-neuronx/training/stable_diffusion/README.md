```
source /opt/aws_neuronx_venv_pytorch_2_7_nxd_training/bin/activate

pip install -U -r requirements.txt

cp modeling_utils.py /opt/aws_neuronx_venv_pytorch_2_7_nxd_training/lib/python3.10/site-packages/diffusers/models/modeling_utils.py

XLA_PARAMETER_WRAPPING_THREADSHOLD=6400 python run.py
```

neuronx-cc compile --framework=XLA /tmp/ubuntu/neuroncc_compile_workdir/51f138b2-bc0c-4b82-8ef6-cb3394f496a5/model.MODULE_4752093806959764898+a8349b9e.hlo_module.pb --output /tmp/ubuntu/neuroncc_compile_workdir/51f138b2-bc0c-4b82-8ef6-cb3394f496a5/model.MODULE_4752093806959764898+a8349b9e.neff --target=trn2 --verbose=DEBUG -O1 --model-type=cnn-training --enable-saturate-infinity