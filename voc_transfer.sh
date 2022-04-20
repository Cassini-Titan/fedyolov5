#!/bin/bash
ClientNum=4

echo "Starting server"
python server.py \
--name transfer_backbone \
--round 20 \
--weights yolov5n.pt \
--data VOC.yaml \
--hyp hyp.VOC.yaml \
--batch-size 1 \
--img 512 \
--device 0 \
--workers 8 \
--half &

sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 1 $ClientNum)
do
    echo "Starting client $i"
    python client_single_gpu.py \
    --name transfer_backbone \
    --id $i \
    --freeze 10 \
    --weights yolov5n.pt \
    --data VOC.yaml \
    --hyp hyp.VOC.yaml \
    --img 512 \
    --batch-size 1 \
    --workers 6 \
    --epochs 5  \
    --nosave \
    --noval \
    --cache \
    --device 0 &
done

# This will allow you to use CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait
