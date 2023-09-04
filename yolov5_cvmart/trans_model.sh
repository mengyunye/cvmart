#!/bin/bash
pip install  onnx onnx-simplifier onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple \
&& python /project/train/src_repo/export.py --weights /project/train/models/exp2/weights/best.pt --include onnx --data /project/train/src_repo/data/my.yaml --simplify --imgsz=480