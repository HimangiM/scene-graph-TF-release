#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"


# arguments when calling train.sh, 1. net, 2. inference iterations, 3. checkpoint directory, 4. gpu
NET=$1
INFERENCE_ITER=$2
EXP_DIR=$3
GPU_ID=$4

#contains all network parameters, train and test parameters
CFG_FILE=experiments/cfgs/sparse_graph.yml 
# pretrained weight in numpy array
PRETRAINED=data/pretrained/coco_vgg16_faster_rcnn_final.npy

# dataset
# db = database (hdf5 format)
# scene graph database in hdf5 format, roi basically
ROIDB=VG-SGG
# roi proposals, rpn = region proposal networks
RPNDB=proposals.h5
# database of images, 1024 pixels
IMDB=imdb_1024.h5
# ?
ITERS=150000

# log
OUTPUT=checkpoints/$EXP_DIR
TF_LOG=checkpoints/$EXP_DIR/tf_logs
rm -rf ${OUTPUT}/logs/
rm -rf ${TF_LOG}
mkdir -p ${OUTPUT}/logs
LOG="$OUTPUT/logs/`date +'%Y-%m-%d_%H-%M-%S'`"

export CUDA_VISIBLE_DEVICES=$GPU_ID

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# the actualy function train_net.py with arguments
# ${} calling the variables declared above
time ./tools/train_net.py --gpu 0 \
  --weights ${PRETRAINED} \     #pretrained weights in numpy array
  --imdb ${IMDB} \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB} \
  --iters ${ITERS} \
  --cfg ${CFG_FILE} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --output ${OUTPUT} \
  --tf_log ${TF_LOG}
