# Train with Shapenet from scratch, using 8 GPUs.
python train.py --outdir=./training-runs --cfg=shapenet \
  --gpus=1 --batch=4 --gamma=0.3 \
  --data_camera_mode=shapenet_chair --data=/home/t-bowenzhang/yiji-project/blob/data/shapenet_chair_render_512/img/03001627 --camera_path=./
