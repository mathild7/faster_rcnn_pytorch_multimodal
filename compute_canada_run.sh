#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=mat-job
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=16             # CPU cores/threads
#SBATCH --gres=gpu:1                   # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL

# Default Command line args
SING_IMG=/home/$USER/projects/def-swasland-ab/$USER/mat_pytorch_5.sif
NUM_ITERATIONS=0
NET_TYPE=''
DATASET='kitti'
ITER=0
SCALE=1
FIXED_BLOCKS=0
PRELOAD=0
EN_FULL_NET=1
EN_FPN=0
EN_ALEATORIC=0
EN_EPISTEMIC=0
UC_SORT_TYPE=''
DATA_DIR=/home/$USER/projects/def-swasland-ab/$USER
CACHE_DIR=/home/$USER/projects/def-swasland-ab/$USER/$DATASET/cache
WEIGHTS_DIR=/home/$USER/weights
WEIGHTS_FILE=''
EXTRA_TAG='default'
DIST=false
TCP_PORT=18888

# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER tools/scripts/${0##*/} [-h]
[--sing_img SING_IMG]
[--iters NUM_ITERATIONS]
[--net_type NET_TYPE]
[--db DATASET]
[--iter ITER]
[--scale SCALE]
[--fixed_blocks FIXED_BLOCKS]
[--preload PRELOAD]
[--en_full_net EN_FULL_NET]
[--en_fpn EN_FPN]
[--data_dir DATA_DIR]
[--cache_dir CACHE_DIR]
[--weights_file WEIGHTS_FILE]
[--epochs EPOCHS]
[--extra_tag 'EXTRA_TAG']
[--tcp_port TCP_PORT]
--sing_img             SING_IMG         Singularity image file              [default=$SING_IMG]
--iters                NUM_ITERATIONS   num iterations to run training      [default=$NUM_ITERATIONS]
--net_type             NET_TYPE         Infos directory                     [default=$NET_TYPE]
--db                   DATASET          which dataset to use                [default=$DATASET]
--iter                 ITER             which training iteration            [default=$ITER]
--scale                SCALE            Scale to run at                     [default=$SCALE]
--fixed_blocks         FIXED_BLOCKS     number of resnet blocks fixed       [default=$FIXED_BLOCKS]
--preload              PRELOAD          Preload parameter                   [default=$PRELOAD]
--en_full_net          EN_FULL_NET      Controls if full net train or RPN   [default=$EN_FULL_NET]
--en_fpn               EN_FPN           Control if FPN is enabled           [default=$EN_FPN]
--en_aleatoric         EN_ALEATORIC     Enable aleatoric uncertainty        [default=$EN_ALEATORIC]
--en_epistemic         EN_EPISTEMIC     Enable epistemic uncertainty        [default=$EN_EPISTEMIC]
--uc_sort_type         UC_SORT_TYPE     Which uncertainty type to sort by   [default=$UC_SORT_TYPE]
--data_dir             DATA_DIR         Directory of dataset                [default=$DATA_DIR]
--cache_dir            CACHE_DIR        Directory of labels cache           [default=$CACHE_DIR]
--weights_dir          WEIGHTS_DIR      Directory of pretrained weights     [default=$WEIGHTS_DIR]
--weights_file         WEIGHTS_FILE     Preload weights file                [default=$WEIGHTS_FILE]
--extra_tag            EXTRA_TAG        Extra experiment tag                [default=$EXTRA_TAG]
--dist                 DIST             Distributed training flag           [default=$DIST]
--tcp_port             TCP_PORT         TCP port for distributed training   [default=$TCP_PORT]
"
}

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    --sing_img)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SING_IMG=$2
            shift
        else
            die 'ERROR: "--iters" requires a non-empty option argument.'
        fi
        ;;
    --iters)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_ITERATIONS=$2
            shift
        else
            die 'ERROR: "--iters" requires a non-empty option argument.'
        fi
        ;;
    --net_type)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NET_TYPE=$2
            shift
        else
            die 'ERROR: "--net_type" requires a non-empty option argument.'
        fi
        ;;
    --db)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATASET=$2
            shift
        else
            die 'ERROR: "--db" requires a non-empty option argument.'
        fi
        ;;
    --iter)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            ITER=$2
            shift
        else
            die 'ERROR: "--iter" requires a non-empty option argument.'
        fi
        ;;        
    --scale)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SCALE=$2
            shift
        else
            die 'ERROR: "--scale" requires a non-empty option argument.'
        fi
        ;;   
    --fixed_blocks)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            FIXED_BLOCKS=$2
            shift
        else
            die 'ERROR: "--fixed_blocks" requires a non-empty option argument.'
        fi
        ;;   
    --preload)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRELOAD=$2
            shift
        else
            die 'ERROR: "--preload" requires a non-empty option argument.'
        fi
        ;; 

    --en_full_net)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EN_FULL_NET=$2
            shift
        else
            die 'ERROR: "--en_full_net" requires a non-empty option argument.'
        fi
        ;; 
    --en_fpn)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EN_FPN=$2
            shift
        else
            die 'ERROR: "--en_fpn" requires a non-empty option argument.'
        fi
        ;; 
    --en_aleatoric)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EN_ALEATORIC=$2
            shift
        else
            die 'ERROR: "--en_aleatoric" requires a non-empty option argument.'
        fi
        ;; 
    --en_epistemic)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EN_EPISTEMIC=$2
            shift
        else
            die 'ERROR: "--en_epistemic" requires a non-empty option argument.'
        fi
        ;; 
    --uc_sort_type)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            UC_SORT_TYPE=$2
            shift
        else
            die 'ERROR: "--uc_sort_type" requires a non-empty option argument.'
        fi
        ;; 
        

    --data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    --cache_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CACHE_DIR=$2
            shift
        else
            die 'ERROR: "--cache_dir" requires a non-empty option argument.'
        fi
        ;;
    --weights_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WEIGHTS_DIR=$2
            shift
        else
            die 'ERROR: "--weights_dir" requires a non-empty option argument.'
        fi
        ;;
    --weights_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WEIGHTS_FILE=$2
            shift
        else
            die 'ERROR: "--weights_file" requires a non-empty option argument.'
        fi
        ;;
        
    -t|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -2|--dist)       # Takes an option argument; ensure it has been specified.
        DIST="true"
        ;;
    -o|--tcp_port)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TCP_PORT=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

echo "Running with the following arguments:
SING_IMG=$SING_IMG
ITERS=$NUM_ITERATIONS
NET_TYPE=$NET_TYPE
DB=$DATASET
ITER=$ITER
SCALE=$SCALE
PRELOAD=$PRELOAD
FIXED_BLOCKS=$FIXED_BLOCKS
EN_FULL_NET=$EN_FULL_NET
EN_FPN=$EN_FPN
EN_ALEATORIC=$EN_ALEATORIC
EN_EPISTEMIC=$EN_EPISTEMIC
UC_SORT_TYPE=$UC_SORT_TYPE
DATA_DIR=$DATA_DIR
CACHE_DIR=$CACHE_DIR
WEIGHTS_DIR=$WEIGHTS_DIR
EXTRA_TAG=$EXTRA_TAG
DIST=$DIST
TCP_PORT=$TCP_PORT
"

# Extract Dataset
# TODO conditional extract
echo "Extracting data"
TMP_DATA_DIR=$SLURM_TMPDIR/data
for file in $DATA_DIR/*.tar; do
    echo "Unziping $file to $TMP_DATA_DIR"
    tar -xvf $file -C $TMP_DATA_DIR
done
echo "Done extracting data"

# Load Singularity
module load singularity/3.5

PROJ_DIR=$PWD

echo "
Running training:
Running SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec
--nv
-B $PROJ_DIR/faster_rcnn_pytorch_multimodal:/workspace/repo
-B $WEIGHTS_DIR:/workspace/weights
-B $SLURM_TMPDIR:/workspace/scratch
$SING_IMG
python /workspace/repo/tools/trainval_net.py
    --cfg_file /Det3D/$CFG_FILE
    --iters $NUM_ITERATIONS 
    --net res101 
    --db $DATASET
    --net_type $NET_TYPE 
    --iter $ITER 
    --scale $SCALE
    --preload $PRELOAD
    --en_full_net $EN_FULL_NET
    --en_fpn $EN_FPN
    --fixed_blocks $FIXED_BLOCKS
    --en_aleatoric $EN_ALEATORIC
    --en_epistemic $EN_EPISTEMIC
    --uc_sort_type $UC_SORT_TYPE
    --data_dir /scratch/data 
    --weights_file /workspace/weights/$WEIGHTS_FILE
    --epochs $EPOCHS
    --extra_tag $EXTRA_TAG
"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec --nv -B $PROJ_DIR/faster_rcnn_pytorch_multimodal:/workspace/repo -B $WEIGHTS_DIR:/workspace/weights -B $CACHE_DIR:/workspace/cache -B $SLURM_TMPDIR:/workspace/data $SING_IMG python /workspace/repo/tools/trainval_net.py --iters $NUM_ITERATIONS 
    --net res101 --db $DATASET --net_type $NET_TYPE --iter $ITER --scale $SCALE --preload $PRELOAD --en_full_net $EN_FULL_NET --en_fpn $EN_FPN --fixed_blocks $FIXED_BLOCKS --en_aleatoric $EN_ALEATORIC --en_epistemic $EN_EPISTEMIC --uc_sort_type $UC_SORT_TYPE --data_dir /workspace/data --cache_dir /workspace/cache --weights_file /workspace/weights/$WEIGHTS_FILE