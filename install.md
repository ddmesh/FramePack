~~~sh
# for windows

# Install python 3.12, not newer because "sentencepiece" is not compatible
# https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe

# in windows powershell
python.exe -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install accelerate
pip install diffusers
pip install transformers
pip install gradio
pip install sentencepiece
pip install pillow
pip install av
pip install numpy
pip install scipy
pip install requests
pip install torchsde
pip install einops
pip install opencv-contrib-python
pip install safetensors


# install sage-attention
# see: https://github.com/woct0rdho/SageAttention/releases/tag/v2.1.1-windows
pip install triton-windows
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp312-cp312-win_amd64.whl
# overwrite python installation for sage attention to add support for sm75 (nvidia rtx2060 super)
#  orignal source at: https://github.com/XUANNISSAN/SageAttention-SM75-path/tree/main
# This usually is installed at c:\users\user\appdata\local\programs\python\python312\lib\site-packages\sageattention
#  



# run and open browser to port which is displayed
FramePack> python.exe .\demo_gradio.py
~~~
