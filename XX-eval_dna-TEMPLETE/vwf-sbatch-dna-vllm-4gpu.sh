#!/bin/bash
#SBATCH --job-name=qwen3_4gpu_dna
#SBATCH --partition=P05
#SBATCH --nodelist=oYOUR_OPENAI_API_KEY
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=120
#SBATCH --time=04:00:00
#SBATCH --output=/home/Competition2025/P05/P05U019/tmp/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P05/P05U019/tmp/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# ===== デバッグ設定 =====
set -x   # 実行するコマンドを逐次表示
set -u   # 未定義変数をエラーにする
set -e   # エラーで即終了
export PS4='+ $(date "+%Y-%m-%d %H:%M:%S") [${BASH_SOURCE##*/}:${LINENO}] '

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_TOKEN= "YOUR_HUGGINGFACE_TOKEN"
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.YOUR_HUGGINGFACE_TOKEN
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- 必要なディレクトリを作成 -----------------------------------------
mkdir -p evaluation_results

#--- vLLM 起動（4GPU）----------------------------------------------
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --reasoning-parser qwen3 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  > vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- 推論 -----------------------------------------------------------
python llm-compe-eval/evaluate_huggingface_models.py \
    --model_name "Qwen/Qwen3-8B" \
    --dataset_path datasets/Instruction/do_not_answer_en.csv \
    --output_dir evaluation_results \
    --use_vllm \
    --max_questions 50 \
    --vllm_base_url http://localhost:8000/v1 > predict.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait
