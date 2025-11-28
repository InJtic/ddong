#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --array=0-3
#SBATCH --partition=gpu2 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ddong_eval
#SBATCH -o ./logs/jupyter.%N.%j.%A.%a.out  # STDOUT 
#SBATCH -e ./logs/jupyter.%N.%j.%A.%a.err  # STDERR

echo "start at:" `date` 
echo "node: $HOSTNAME" 
echo "jobid: $SLURM_JOB_ID, array_task_id: $SLURM_ARRAY_TASK_ID"

module unload CUDA/11.2.2 
module load cuda/11.8.0

set +u
DDONG_ROOT=/gpfs/home1/jtic0524/ddong
export PYTHONPATH="$PYTHONPATH:$DDONG_ROOT/lmms-eval"
set -u

cd "$DDONG_ROOT/lmms-eval"



case $SLURM_ARRAY_TASK_ID in
   0)
      MODEL_TYPE="internvl2"
      MODEL_ARGS="pretrained=OpenGVLab/InternVL3_5-8B"
      ;;
   1)
      MODEL_TYPE="qwen3_vl"
      MODEL_ARGS="pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=256*28*28,max_pixels=1280*28*28"
      ;;
   2)
      # lmms-lab/LLaVA-Video-7B-Qwen2
      # LLaVA-Video는 전용 로더(llava_video)와 conv_template 설정이 중요합니다.
      MODEL_TYPE="llava_vid"
      MODEL_ARGS="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32"
      ;;
   3)
      MODEL_TYPE="phi4_multimodal" 
      MODEL_ARGS="pretrained=microsoft/Phi-4-multimodal-instruct,trust_remote_code=True"
      ;;
esac

OUTPUT_DIR="$DDONG_ROOT/output/results"

python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('Current CUDA device:', torch.cuda.current_device()); print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))"

accelerate launch --num_processes 1 \
   --main_process_port $((29500 + $SLURM_ARRAY_TASK_ID)) \
   -m lmms_eval \
   --model $MODEL_TYPE \
   --model_args $MODEL_ARGS \
   --tasks ddong_bench \
   --batch_size 1 \
   --log_samples \
   --output_path $OUTPUT_DIR