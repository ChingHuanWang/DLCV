#!bin/bash
wget https://www.dropbox.com/s/dv1j0yrtevof2nh/model.pth?dl=0 -O model.pth
python3 hw1_1.py --mode test --test_repo $1 --csv_path $2
