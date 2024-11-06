log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 val.py --model "$1" > "$2" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 val.py --model $1" > /dev/null; do
        sleep 10
    done
}

task() {
    start_background_task "$1" "$2"
    wait_for_completion "$1"
}
log_dir="./logs"
log_directory_check "$log_dir"

task runs/detect/deco-yolo8n/weights/best.pt ./logs/val/deco-yolo8n.log
task runs/detect/deco-yolo10n/weights/best.pt ./logs/val/deco-yolo10n.log
task runs/detect/deco-yolo11n/weights/best.pt ./logs/val/deco-yolo11n.log