thre_val=0.090
model="CPSP_MMIL_Net"
model_save_dir="models"
lambda_v=1.0
ratio=0.9
eta=0.2
bs=16
data_root="please change this to your path of AVVP dataset"


python main_avvp_psp.py --mode retrain 
--audio_dir ${data_root}/feats/vggish/ \
--video_dir ${data_root}/feats/res152/ \
--st_dir ${data_root}/feats/r2plus1d_18 \
--threshold ${thre_val} \
--checkpoint ${model} \
--vcon_flag \
--lambda_v ${lambda_v} \
--ratio ${ratio} \
--eta ${eta} \
--batch-size ${bs} \
--model_save_dir ${model_save_dir}
