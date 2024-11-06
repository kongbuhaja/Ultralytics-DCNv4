log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 train.py --model "$1" > "$2/$1" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 train.py --model $1" > /dev/null; do
        sleep 30
    done
}

task() {
    start_background_task "$1" "$2" "$3"
    wait_for_completion "$1"
}

log_dir="./logs"
log_directory_check "$log_dir"

task deco-yolov8n "$log_dir"
task deco-yolov10n "$log_dir"
task deco-yolov11n "$log_dir"
