#!/bin/bash
ClientNum=1

echo "Starting server"
python server.py \
--name local_adapt \
--round 1 \
--weights ./checkpoints/client4/last.pt \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-med.yaml \
--batch-size 128 \
--img 640 \
--device 0 \
--workers 8 \
--half &

sleep 3  # Sleep for 3s to give the server enough time to start

# client1:no freeze b=32, e=4
python client_single_gpu.py \
--name local_adapt \
--id 1 \
--weights ./checkpoints/client4/last.pt \
--data ./data/kitti.yaml \
--hyp ./data/hyps/hyp.scratch-med.yaml \
--img 640 \
--batch-size 64 \
--workers 4 \
--epochs 20  \
--nosave \
--device 0 &

# This will allow you to use CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait
