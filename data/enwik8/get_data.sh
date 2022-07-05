#!/bin/bash
# from https://github.com/facebookresearch/adaptive-span/blob/main/get_data.sh

if [[ ! -d 'data/enwik8' ]]; then
  mkdir -p data/enwik8
  cd data/enwik8
  echo "Downloading enwik8 data ..."
  wget --continue http://mattmahoney.net/dc/enwik8.zip
  wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
  python3 prep_enwik8.py
  cd ../..
fi