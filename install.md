~~~sh
# for windows

# Install python 3.12, not newer because "sentencepiece" is not compatible
# https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe

# in windows powershell
python.exe -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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
pip install triton-windows

# following have versions conflicts or run time errors 
# pip install xformers
# pip install flash_attn
# pip install sageattention

~~~
