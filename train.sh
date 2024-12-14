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
    nohup python3 train.py --model "$1" --cpus "$2" --gpus "$3" > /dev/null 2>&1 & #"$4/$1.log" 2>&1 &
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
# task deco-yolo11n "$log_dir" 

# 2,724,448     2,775,520     2,624,080

task deco-yolo8n "0-15" 0 "$log_dir" # 2,731,184             2,759,504 | 2,751,872 2,759,504
task deco-yolo10n "16-31" 1 "$log_dir" # 2,782,256 2,786,992  2,810,576 | 2,802,944 2,810,576
# task deco-yolo11n "0-9" 0 "$log_dir" # 2,660,752 2,633,776 2,657,552 | 2,665,456 2,657,552

task psd-yolo10n "32-47" 2 "$log_dir" # 2,780,720 2,781,896  2,809,856 | 2,818,104 2,810,576
# task c2psd-yolo11n "16-31" 1 "$log_dir" # 2,629,280 2,630,456  2,658,432 | 2,666,664 2,670,384

task deco-psd-yolo10n "48-64" 3 "$log_dir" # 2,787,456 2,793,368       2,844,928 | 2,896,328 2,856,880
# task deco-c2psd-yolo11n "48-63" 3 "$log_dir" # 2,665,952 2,640,152     2,691,904 | 2,708,040 2,703,856


