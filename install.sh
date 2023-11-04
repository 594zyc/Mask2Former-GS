# conda create --name m2f python=3.10 ninja
# source activate m2f

pip install torch torchvision
pip install -U opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -r requirements.txt

cd detectron2
pip install -e .
cd ..

cd mask2former/modeling/pixel_decoder/ops
sh make.sh