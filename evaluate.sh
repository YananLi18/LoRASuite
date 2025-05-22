start_time=$(date +%s%N)

start_index=${7:-0}
elements=('AddSub' 'MultiArith' 'SingleEq' 'gsm8k' 'AQuA' 'mawps' 'SVAMP')
# elements=('mawps' 'SVAMP')
new_elements=("${elements[@]:start_index}")

for element in "${new_elements[@]}"; do
    if [[ "$6" == "true" ]]; then
        echo "Evaluate $element with LoRA"
        CUDA_VISIBLE_DEVICES=$1 python evaluate.py --dataset $element --model $2 --base_model $3 --lora_weights $4 --batch_size $5 --with_lora
    else
        echo "Evaluate $element without LoRA"
        CUDA_VISIBLE_DEVICES=$1 python evaluate.py --dataset $element --model $2 --base_model $3 --lora_weights $4 --batch_size $5
    fi
done

end_time=$(date +%s%N)
elapsed_time=$(awk "BEGIN {printf \"%.3f\", ($end_time - $start_time) / 1000000000}")
echo "Script execution time: ${elapsed_time} seconds"
