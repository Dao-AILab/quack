# Sourced where nvidia-smi sees the devices available to this job. The largest
# fp32 RMSNorm case needs ~45 GiB. Wait briefly if another tenant gets there first.
if ! [[ "$GPU_WAIT_SECONDS" =~ ^[0-9]+$ && "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo '::error::gpu_wait_seconds must be non-negative and gpu_poll_seconds must be positive'
    exit 2
fi
deadline=$((SECONDS + GPU_WAIT_SECONDS))
while :; do
    FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F', ' '$2 >= 50000 {print $1}' | paste -sd,) || FREE_GPUS=''
    [[ -z "$FREE_GPUS" ]] || break
    if (( SECONDS >= deadline )); then
        echo "::error::No GPUs with >= 50GB free after ${GPU_WAIT_SECONDS}s"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits || true
        exit 1
    fi
    echo "Waiting for a GPU with >= 50GB free; retrying in ${GPU_POLL_SECONDS}s"
    sleep "$GPU_POLL_SECONDS"
done
NUM_GPUS=$(tr ',' '\n' <<< "$FREE_GPUS" | wc -l)
export FREE_GPUS NUM_GPUS
echo "Running tests on GPUs: $FREE_GPUS"
