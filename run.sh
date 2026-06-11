export PYTHONPATH=$(cd "$(dirname "$0")" && pwd):$PYTHONPATH

export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT="A character with blond hair, wearing a blue tunic, white pants, and brown boots, stands on a cobblestone path, facing away from the viewer. They hold a large shield with a stylized ""Z"" on it.  A brown horse stands in a wooden and stone stable to the left.  The stable has a wooden roof supported by wooden beams.  A wooden fence runs along the back of the stable.  The path extends towards rolling green hills under a bright blue sky.  A small wooden sign is visible on the stable's roof.  The scene is brightly lit, suggesting a daytime setting."

IMAGE_PATH=./assets/img/1.png # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
OUTPUT_PATH=./outputs/
MODEL_PATH=/irip/huangyu_2026/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038                   # Path to pretrained hunyuanvideo-1.5 model
AR_ACTION_MODEL_PATH=/irip/huangyu_2026/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 autoregressive checkpoints
BI_ACTION_MODEL_PATH=/irip/huangyu_2026/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/bidirectional_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 bidirectional checkpoints
AR_DISTILL_ACTION_MODEL_PATH=/irip/huangyu_2026/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_distilled_action_model/diffusion_pytorch_model.safetensors # Path to our HY-World 1.5 autoregressive distilled checkpoints
POSE='w-31'                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
NUM_FRAMES=125
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=1 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# inference with bidirectional model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

# inference with autoregressive model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_ACTION_MODEL_PATH \
#   --few_step false \
#   --width $WIDTH \
#   --height $HEIGHT \
#   --model_type 'ar'

# inference with autoregressive distilled model
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
  --few_step true \
  --num_inference_steps 4 \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar' \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
  --transformer_resident_ar_rollout true \
  --use_sdtm true \
  --sdtm_ratio 0.2 \
  --sdtm_deviation 0.05 \
  --sdtm_sx 4 \
  --sdtm_sy 5 \
  --sdtm_switch_step 1 \
  --sdtm_protect_steps_frequency -1 \
  --sdtm_auto_window true \
  --sdtm_verbose true
