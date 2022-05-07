./centerpoint \
--pfeOnnxPath ../tools/pfe.onnx \
--rpnOnnxPath ../tools/rpn.onnx \ 
--pfeEnginePath  ../tools/pfe_fp.engine \
--rpnEnginePath  ../tools/rpn_fp.engine \
--savePath ../results \ 
--filePath /mnt/data/waymo_opensets/val/points \
--loadEngine true \
--fp16 true
