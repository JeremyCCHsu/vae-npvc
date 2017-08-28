bash download.sh
pip install -U pip
pip install -r requirements.txt
python analysze.py
python build.py
python main.py \
  --model ConvVAE \
  --trainer VAETrainer \
  --architecture architecture-vae-vcc2016.json
python convert.py \
  --src SF1 \
  --trg TM3 \
  --model ConvVAE \
  --checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
  --file_pattern "./dataset/vcc2016/bin/Testing Set/{}/*.bin"
echo "Please find your results in `./logdir/output/[timestamp]`"
