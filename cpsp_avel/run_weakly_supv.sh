SEED=123
DATASET="ave"
if [ $DATASET == "ave" ]
then
    CATEGORY_NUM=28
elif [ $DATASET == "vggsound" ]
then 
    CATEGORY_NUM=141
fi
MODEL_NAME='cpsp'
BS=64
THRESHOLD=0.095
NUM_EPOCHS=45
VIS_FEA_TYPE='vgg'
VIS_FEA_DIR="your visual feature path" # should be modified


python weakly_supv_main.py \
--lr 0.001 \
--clip_gradient 0.1 \
--seed ${SEED} \
--dataset_name ${DATASET} \
--model ${MODEL_NAME} \
--snapshot_pref "./TestExps/${DATASET}/WeaklySupv/exp_${MODEL_NAME}_seed${SEED}_bs${BS}" \
--category_num ${CATEGORY_NUM} \
--vis_maps_path ${VIS_FEA_DIR} \
--batch_size ${BS} \
--test_batch_size ${BS} \
--threshold_value ${THRESHOLD} \
--n_epoch ${NUM_EPOCHS} \
--print_freq 200 \
--val_print_freq 4 \
--eval_freq 1 \
--lambda_vcon 1 \
--vcon_flag \
--margin 0.6 \
--neg_num 4 \
# --vcon_infoNCE_flag \
# --eta 0.1
# --lambda_sscon 1 \
