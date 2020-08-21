#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=mat-job
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=16             # CPU cores/threads
#SBATCH --gres=gpu:1                   # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-3%1   # 3 is the number of jobs in the chain

# Default Command line args
DATA_DIR=/home/$USER/projects/def-swasland-ab/$USER
CACHE_DIR=/home/$USER/projects/def-swasland-ab/$USER/cache
DATASET='waymo'
SING_IMG=/home/$USER/projects/def-swasland-ab/$USER/mat_pytorch_5.sif
BATCH_SIZE=4
PRETRAINED_MODEL=None
EPOCHS=80
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
--iters                ITERS            num iterations to run training      [default=$DATA_DIR]
--net_type             NET_TYPE         Infos directory                     [default=$NET_TYPE]
--db                   DATASET          which dataset to use                [default=$DATASET]
--iter                 ITER             which training iteration            [default=$ITER]
--scale                SCALE            Scale to run at                     [default=$SCALE]
--preload              PRELOAD          Preload parameter                   [default=$PRELOAD]
--en_full_net          EN_FULL_NET      Controls if full net train or RPN   [default=$EN_FULL_NET]
--en_fpn               EN_FPN           Control if FPN is enabled           [default=$EN_FPN]
--en_aleatoric         en_aleatoric     Enable aleatoric uncertainty        [default=$EN_FPN]
--en_epistemic         en_epistemic     Enable epistemic uncertainty        [default=$EN_FPN]
--uc_sort_type         UC_SORT_TYPE     Which uncertainty type to sort by   [default=$UC_SORT_TYPE]
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
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    -i|--infos_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            INFOS_DIR=$2
            shift
        else
            die 'ERROR: "--infos_dir" requires a non-empty option argument.'
        fi
        ;;
    -s|--sing_img)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            INFOS_DIR=$2
            shift
        else
            die 'ERROR: "--sing_img" requires a non-empty option argument.'
        fi
        ;;
    -b|--batch_size)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BATCH_SIZE=$2
            shift
        else
            die 'ERROR: "--batch_size" requires a non-empty option argument.'
        fi
        ;;
    -c|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG_FILE=$2
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -p|--pretrained_model)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL=$2
            shift
        else
            die 'ERROR: "--pretrained_model" requires a non-empty option argument.'
        fi
        ;;
    -e|--epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EPOCHS=$2
            shift
        else
            die 'ERROR: "--pretrained_model" requires a non-empty option argument.'
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
DATA_DIR=$DATA_DIR
INFOS_DIR=$INFOS_DIR
SING_IMG=$SING_IMG
CFG_FILE=$CFG_FILE
BATCH_SIZE=$BATCH_SIZE
PRETRAINED_MODEL=$PRETRAINED_MODEL
EPOCHS=$EPOCHS
EXTRA_TAG=$EXTRA_TAG
DIST=$DIST
TCP_PORT=$TCP_PORT
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

# Extract Dataset
echo "Extracting data"
TMP_DATA_DIR=$SLURM_TMPDIR/data
for file in $DATA_DIR/*.zip; do
    echo "Unziping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting data"

# Extract dataset infos
echo "Extracting dataset infos"
for file in $INFOS_DIR/*.zip; do
    echo "Unziping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting dataset infos"

# Load Singularity
module load singularity/3.5

PROJ_DIR=$PWD
DET3D_BINDS=""
for entry in $PROJ_DIR/det3d/*
do
    name=$(basename $entry)
    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        DET3D_BINDS+="--bind $entry:/Det3D/det3d/$name "
    fi
done

if [ $DIST != "true" ]
then
    echo "
Running training:
Running SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec
--nv
--bind $PROJ_DIR/checkpoints:/Det3D/checkpoints
--bind $PROJ_DIR/output:/Det3D/output
--bind $PROJ_DIR/tests:/Det3D/tests
--bind $PROJ_DIR/tools:/Det3D/tools
--bind $TMP_DATA_DIR:/Det3D/$INFOS_DIR
$DET3D_BINDS
$SING_IMG
python /Det3D/tools/train.py
    --cfg_file /Det3D/$CFG_FILE
    --workers $SLURM_CPUS_PER_TASK
    --batch_size $BATCH_SIZE
    --pretrained_model $PRETRAINED_MODEL
    --epochs $EPOCHS
    --extra_tag $EXTRA_TAG
"
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec \
        --nv \
        --bind $PROJ_DIR/checkpoints:/Det3D/checkpoints \
        --bind $PROJ_DIR/output:/Det3D/output \
        --bind $PROJ_DIR/tests:/Det3D/tests \
        --bind $PROJ_DIR/tools:/Det3D/tools \
        --bind $TMP_DATA_DIR:/Det3D/$INFOS_DIR \
        $DET3D_BINDS \
        $SING_IMG \
        python /Det3D/tools/train.py \
            --cfg_file /Det3D/$CFG_FILE \
            --workers $SLURM_CPUS_PER_TASK \
            --batch_size $BATCH_SIZE \
            --pretrained_model /Det3D/$PRETRAINED_MODEL \
            --epochs $EPOCHS \
            --extra_tag $EXTRA_TAG
else
    echo "
Running training:
Running SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1 singularity exec
--nv
--bind $PROJ_DIR/checkpoints:/Det3D/checkpoints
--bind $PROJ_DIR/output:/Det3D/output
--bind $PROJ_DIR/tests:/Det3D/tests
--bind $PROJ_DIR/tools:/Det3D/tools
--bind $TMP_DATA_DIR:/Det3D/$INFOS_DIR
$DET3D_BINDS
$SING_IMG
python -m torch.distributed.launch
    --nproc_per_node=2
    /Det3D/tools/train.py
        --cfg_file /Det3D/$CFG_FILE
        --workers $SLURM_CPUS_PER_TASK
        --batch_size $BATCH_SIZE
        --pretrained_model $PRETRAINED_MODEL
        --epochs $EPOCHS
        --extra_tag $EXTRA_TAG
        --launcher pytorch
        --sync_bn
        --tcp_port $TCP_PORT
"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1 singularity exec \
    --nv \
    --bind $PROJ_DIR/checkpoints:/Det3D/checkpoints \
    --bind $PROJ_DIR/output:/Det3D/output \
    --bind $PROJ_DIR/tests:/Det3D/tests \
    --bind $PROJ_DIR/tools:/Det3D/tools \
    --bind $TMP_DATA_DIR:/Det3D/$INFOS_DIR \
    $DET3D_BINDS \
    $SING_IMG \
    python -m torch.distributed.launch \
    --nproc_per_node=2 \
    /Det3D/tools/train.py \
        --cfg_file /Det3D/$CFG_FILE \
        --workers $SLURM_CPUS_PER_TASK \
        --batch_size $BATCH_SIZE \
        --pretrained_model $PRETRAINED_MODEL \
        --epochs $EPOCHS \
        --extra_tag $EXTRA_TAG \
        --launcher pytorch \
        --sync_bn \
        --tcp_port $TCP_PORT
fi
echo "Done training"

echo "
Running evaluation:
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec
--nv
--bind $PROJ_DIR/checkpoints:/Det3D/checkpoints
--bind $PROJ_DIR/output:/Det3D/output
--bind $PROJ_DIR/tests:/Det3D/tests
--bind $PROJ_DIR/tools:/Det3D/tools
--bind $TMP_DATA_DIR:/Det3D/$INFOS_DIR
$DET3D_BINDS
$SING_IMG
python /Det3D/tools/test.py
    --cfg_file /Det3D/$CFG_FILE
    --workers $SLURM_CPUS_PER_TASK
    --batch_size $BATCH_SIZE
    --extra_tag $EXTRA_TAG
    --eval_all
"

# Run eval. Make sure to save singularity image in singularity/det3d-singularity.simg
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec \
    --nv \
    --bind $PROJ_DIR/checkpoints:/Det3D/checkpoints \
    --bind $PROJ_DIR/output:/Det3D/output \
    --bind $PROJ_DIR/tests:/Det3D/tests \
    --bind $PROJ_DIR/tools:/Det3D/tools \
    --bind $TMP_DATA_DIR:/Det3D/$INFOS_DIR \
    $DET3D_BINDS \
    $SING_IMG \
    python /Det3D/tools/test.py \
        --cfg_file /Det3D/$CFG_FILE \
        --workers $SLURM_CPUS_PER_TASK \
        --batch_size $BATCH_SIZE \
        --extra_tag $EXTRA_TAG \
        --eval_all
echo "Done evaluation"