export FREESURFER_HOME=./sw/freesurfer/
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export CUDA_VISIBLE_DEVICES=0

./sw/run_fastsurfer.sh --t1 $1 \
                    --seg_only \
                    --sd $2 \
                    --sid $3 \
                    --threads 8 \
                    --device cuda:0 \
                    --fsaparc \
                    --fs_license $FREESURFER_HOME/license.txt \
                    --py python \
                    --no_biasfield \
                    --no_hypothal \
                    --no_cereb
                    


