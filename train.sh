log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    echo "$1"
    nohup python3 train.py --model "$1" --cpus "$2" --gpus "$3" > "$4/$1.log" 2>&1 &
    sleep 10
}

wait_for_completion() {
    while pgrep -f "python3 train.py --model $1 --cpus $2 --gpus $3" > /dev/null; do
        sleep 30
    done
}

task() {
    start_background_task "$1" "$2" "$3" "$4"
    # wait_for_completion "$1"
}

log_dir="./logs"
log_directory_check "$log_dir"

# task deco-yolo8n "$log_dir"
# task deco-yolo10n "$log_dir"
# task deco-yolo11n "$log_dir" # dc3k2-dbottle, dconv-c2

# 2,724,448     2,775,520     2,624,080


# task deco-yolo8n "0-15" 0 "$log_dir" # 2,731,184
# task deco-yolo10n "0-15" 0 "$log_dir" # 2,782,256 2,786,992
# task deco-yolo11n "16-31" 1 "$log_dir" # 2,660,752 2,633,776

# task psd-yolo10n "32-47" 2 "$log_dir" # 2,780,720 2,781,896
task c2psd-yolo11n "48-63" 3 "$log_dir" # 2,629,280 2,630,456

# task deco-psd-yolo10n "$log_dir" # 2,787,456 2,793,368
# task deco-c2psd-yolo11n "$log_dir" # 2,665,952 2,640,152
