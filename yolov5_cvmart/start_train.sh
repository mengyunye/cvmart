#!/bin/bash

project_root_dir=/project/train/src_repo
log_file=/project/train/log/log.txt
tb_path=/project/train/tensorboard
plt_path=/project/train/models/exp
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /project/train/src_repo/requirements.txt \
&& echo "Preparing..." \
&& echo "Converting dataset..." \
&& python3 -u ${project_root_dir}/convert_dataset.py | tee -a ${log_file} \
&& echo "Start training..." \
&& cd ${project_root_dir} && python3 -u train.py --epochs=50 --weights=/project/train/models/exp2/weights/best.pt --hyp=hyp.scratch-med.yaml | tee -a ${log_file} \
&& echo "Copy png..." \
&& cp ${plt_path}/F1_curve.png ${tb_path}/ \
&& cp ${plt_path}/labels.jpg ${tb_path}/ \
&& cp ${plt_path}/labels_correlogram.jpg ${tb_path}/ \
&& cp ${plt_path}/results.png ${tb_path}/ \
&& cp ${plt_path}/confusion_matrix.png ${tb_path}/ \
&& echo "Done" 
# && echo "Remove result-graphs file" \
# && rm  ${plt_path}/* \
# && echo "Copy font..." \
# && cp ${project_root_dir}/Arial.ttf /project/.config/Ultralytics/Arial.ttf | tee -a ${log_file} \
