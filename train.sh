log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 test.py --model "$1" --data "$2" > "$3/$1_$2.log" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 test.py --model $1 --data $2" > /dev/null; do
        sleep 10
    done
}

task() {
    start_background_task "$1" "$2" "$3"
    wait_for_completion "$1" "$2"
}

log_dir="./logs"
log_directory_check "$log_dir"
# task yolov10n coco "$log_dir"
# task dyolov10n coco "$log_dir"

export CUDA_VISIBLE_DEVICES=0
nohup python3 test.py --model dyolov8n --gpus 0 > ./logs/dyolov8n.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=1
nohup python3 test.py --model dyolov10n --gpus 1 > ./logs/dyolov10n.log 2>&1 &
sleep 10
export CUDA_VISIBLE_DEVICES=2
nohup python3 test.py --model dyolov11n --gpus 2 > ./logs/dyolov11n.log 2>&1 &