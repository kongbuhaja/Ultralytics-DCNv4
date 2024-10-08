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
task dyolov10n coco "$log_dir"