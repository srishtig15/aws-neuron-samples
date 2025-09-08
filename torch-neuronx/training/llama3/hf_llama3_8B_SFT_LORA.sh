source /opt/aws_neuronx_venv_pytorch_2_7_nxd_training/bin/activate

export DATA_DIR=~/example_datasets/llama3_8b
mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
aws s3 cp s3://neuron-s3/training_datasets/llama/sft/training.jsonl .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/llama/sft/validation.jsonl .  --no-sign-request

wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/config.json -O ~/config.json

pip install -U -r requirements.txt

hf auth login  # Input your huggignface token

python cache_hf_model.py

wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/refs/heads/main/examples/checkpoint_converter_scripts/checkpoint_converter.py
python checkpoint_converter.py --model_style hf --hw_backend trn2 --hf_model_name meta-llama/Meta-Llama-3-8B --output_dir /home/ubuntu/converted_hf_style_hf_to_nxdt_tp8pp4/ --save_xser True --config /home/ubuntu/llama3-8B_hf_weights/config.json --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 --qkv_linear True --convert_from_full_state

git clone https://github.com/aws-neuron/neuronx-distributed-training ~/neuronx-distributed-training
cd ~/neuronx-distributed-training/examples
export COMPILE=1
./train.sh

export COMPILE=0
./train.sh
