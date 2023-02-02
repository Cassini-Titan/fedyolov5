#!/bin/bash
ClientNum=4

echo "Starting server"
python server.py \
--name kitti_transfer_nofreeze_default \
--round 40 \
--weights yolov5s.pt \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-med.yaml \
--batch-size 32 \
--img 640 \
--device 0 \
--workers 4 \
--half &

sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 $ClientNum)
do
    echo "Starting client $i"
    python client_single_gpu.py \
    --name kitti_transfer_nofreeze_default \
    --id $i \
    --weights yolov5s.pt \
    --data ./data/kitti.yaml \
    --hyp ./data/hyps/hyp.scratch-med.yaml \
    --img 640 \
    --batch-size 20 \
    --workers 4 \
    --epochs 2  \
    --nosave \
    --noval \
    --device 0 &
done

# This will allow you to use CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait
