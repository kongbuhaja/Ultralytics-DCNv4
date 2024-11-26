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
    nohup python3 train.py --model "$1" > "$2/$1.log" 2>&1 &
    sleep 10
}

wait_for_completion() {
    while pgrep -f "python3 train.py --model $1" > /dev/null; do
        sleep 30
    done
}

task() {
    start_background_task "$1" "$2" 
    wait_for_completion "$1"
}

log_dir="./logs"
log_directory_check "$log_dir"

# task deco-yolo8n "$log_dir"
# task deco-yolo10n "$log_dir"
# task deco-yolo11n "$log_dir" # dc3k2-dbottle, dconv-c2

# 2,724,448     2,775,520     2,624,080


task deco-yolo8n "$log_dir" #2,700,464  2,753,600
task deco-yolo10n "$log_dir" #2,751,536  2,804,672
task deco-yolo11n "$log_dir" #2,588,432  2,639,408

task psd-yolo10n "$log_dir" #2,775,384  2,775,384
task c2psd-yolo11n "$log_dir" #2,623,944

task deco-psd-yolo10n "$log_dir" #2,751,400  2,804,536
task deco-c2psd-yolo11n "$log_dir" #2,588,296  2,639,272

# wait_for_completion deco-yolo11n
# start_background_task deco-yolo11n "$log_dir"
