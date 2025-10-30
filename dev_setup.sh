cd ..
git clone https://github.com/facebookresearch/sam2.git

cd sam2
pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
pip install .

pip install jupyter matplotlib opencv-python
