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
THRESHOLD=0.099
NUM_EPOCHS=45
VIS_FEA_TYPE='vgg'
VIS_FEA_DIR="your visual feature path" # should be modified



python fully_supv_main.py \
--lr 0.001 \
--clip_gradient 0.1 \
--seed ${SEED} \
--dataset_name ${DATASET} \
--model ${MODEL_NAME} \ 
--snapshot_pref "./TestExps/${DATASET}/FullySupv/exp_${MODEL_NAME}_${VIS_FEA_TYPE}_seed${SEED}_bs${BS}" \
--category_num ${CATEGORY_NUM} \
--vis_fea_type ${VIS_FEA_TYPE} \
--vis_maps_path ${VIS_FEA_DIR} \
--batch_size ${BS} \
--test_batch_size ${BS} \
--threshold_value ${THRESHOLD}  \
--n_epoch ${NUM_EPOCHS} \
--print_freq 200 \
--eval_freq 1 \
--avps_flag \
--lambda_avps 100 \
--vcon_flag \
--lambda_vcon 1 \
--scon_flag \
--lambda_scon 0.01 \
# --lambda_sscon 1 \
