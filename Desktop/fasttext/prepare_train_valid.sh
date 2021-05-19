#!/bin/bash
echo "ConnectCom Classification v4.1 - Site Mercado"
echo "Split text in train and validate"

head -n 130000 $1 > $1.train
head -n 1 $1 > $1.valid
tail -n 20000 $1 >> $1.valid


