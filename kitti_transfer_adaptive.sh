#!/bin/bash
ClientNum=4

echo "Starting server"
python server.py \
--name kitti_transfer_adaptive_default \
--round 30 \
--weights yolov5s.pt \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-med.yaml \
--batch-size 64 \
--img 640 \
--device 0 \
--workers 4 \
--half &

sleep 3  # Sleep for 3s to give the server enough time to start

# client1:no freeze b=32, e=4
python client_single_gpu.py \
--name kitti_transfer_adaptive_default \
--id 1 \
--weights yolov5s.pt \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-high.yaml \
--img 640 \
--batch-size 32 \
--workers 2 \
--epochs 2  \
--nosave \
--noval \
--device 0 &

# client2:freeze backbone b=24 e=3
python client_single_gpu.py \
--name kitti_transfer_adaptive_default \
--id 2 \
--weights yolov5s.pt \
--freeze 10 \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-med.yaml \
--img 640 \
--batch-size 24 \
--workers 2 \
--epochs 2 \
--nosave \
--noval \
--device 0 &

# client3:freeze backbone b=24 e=3
python client_single_gpu.py \
--name kitti_transfer_adaptive_default \
--id 3 \
--weights yolov5s.pt \
--freeze 10 \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-med.yaml \
--img 640 \
--batch-size 20 \
--workers 2 \
--epochs 2  \
--nosave \
--noval \
--device 0 &

# client4:freeze all b=16 e=2
python client_single_gpu.py \
--name kitti_transfer_adaptive_default \
--id 4 \
--weights yolov5s.pt \
--freeze 24 \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-low.yaml \
--img 640 \
--batch-size 16 \
--workers 2 \
--epochs 2  \
--nosave \
--noval \
--device 0 &

# This will allow you to use CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait
