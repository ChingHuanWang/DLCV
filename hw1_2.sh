#!bin/bash
wget https://www.dropbox.com/s/g6lrqy0eh3oap10/model.ckpt?dl=0 -O model.ckpt
python3 hw1_2.py --mode test --test_repo $1 --output_repo $2
