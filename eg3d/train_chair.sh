/mnt/blob3/azcopy copy "https://facevcstandard.blob.core.windows.net/v-bowenz/data/shapenet_chair_render_512/?sv=2021-04-10&st=2023-08-24T01%3A40%3A57Z&se=2024-08-25T01%3A40%3A00Z&sr=c&sp=racwl&sig=PH9t2116cYbUiqtxY40CpwSOLyvolW0HOH4ss1nmYiI%3D" "./" --overwrite=prompt --check-md5 FailIfDifferent --from-to=BlobLocal --recursive --log-level=INFO;
cp /mnt/blob/data/labels_shapenet_chair.npz ./

export PATH=~/.local/bin:$PATH
export PATH=/usr/local/cuda/bin:$PATH

# Train with Shapenet from scratch, using 8 GPUs.
python train.py --outdir=/mnt/blob/output_0224/eg3d_512_shapenet_chair --cfg=shapenet \
  --gpus=8 --batch=32 --gamma=0.3 \
  --data_camera_mode=shapenet_chair --data=./shapenet_chair_render_512/img/03001627 --camera_path=./


# python train.py --outdir=./training-runs --cfg=shapenet \
#   --gpus=1 --batch=4 --gamma=0.3 \
#   --data_camera_mode=shapenet_chair --data=./shapenet_chair_render_512/img/03001627 --camera_path=./