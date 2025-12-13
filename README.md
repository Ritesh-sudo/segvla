# SegVLA 

## Setup 

1. Install custom lerobot with segvla policy:

```python
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
conda install -c huggingface transformers
pip install num2words

git clone https://github.com/ishandotsh/segvla.git
cd segvla/lerobot
pip install -e ".[libero]"
```

2. Train SegVLA

LIBERO-Spatial 10 with augmented segmentation data is uploaded to HuggingFace at ishandotsh/libero_10_with_m2f_seg

replace `ishandotsh/segvla_libero_m2f_10k` with your own 

the eval_* and env* options can be removed for training

```bash

mkdir outputs

lerobot-train \
  --policy.type=segvla \
  --dataset.repo_id=ishandotsh/libero_10_with_m2f_seg \
  --output_dir=./outputs/segvla_libero_spatial \
  --policy.repo_id=ishandotsh/segvla_libero_m2f_10k \
  --batch_size=16 \
  --steps=10000 \
  --policy.device=cuda \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval_freq=2000

```

3. Final Evaluation

This will run eval on our trained model

```bash

mkdir eval_logs

python src/lerobot/scripts/lerobot_eval.py \
  --output_dir=./eval_logs/segvla_10k   \
  --env.type=libero   \
  --env.task=libero_spatial   \
  --eval.batch_size=1   \
  --eval.n_episodes=10   \
  --policy.path=ishandotsh/segvla_libero_m2f_10k   \
  --policy.n_action_steps=10   \
  --env.max_parallel_tasks=1
```