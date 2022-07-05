下载数据集：

```
wget --continue http://mattmahoney.net/dc/enwik8.zip
```

or：

```
wget --continue https://data.deepai.org/enwik8.zip
```

切换到enwik8.zip所在文件夹，预处理：

```
python prep_enwik8.py
```

trev处理：

```
trev-preprocess --only-source --trainpref ~/data/enwik8/train.txt \
    --validpref ~/data/enwik8/valid.txt --testpref ~/data/enwik8/test.txt \
    --destdir ~/data/enwik8/data-bin/ --joined-dictionary --workers 5
```

