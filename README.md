# 🧞‍♂️🎅🦹‍♂️RLGAN-project-MAADM-UPM 👽🤖👹
Neuroevolution to learn the Lunar Lander from Gymnasium and a GAN to learn to color images. Subject from the ML and BD master´s degree of UPM.

# 🧙‍♂️ Create ".venv_torch"
You need correct CUDA, CUDA Toolkit and cudNN versions installed according to your GPU.

* Python 3.10 version
* Pytorch 2.5 version

If installing with "requirements_torch.txt" there´s an error with torch:
1. Install everything except "torch", torchaudio" and "torchvision" (comment them):
   
    ```pip3 install -r requirements_torch.txt```

2. Verify installing specific gymnasium dependencies:

    ```pip3 install gymnasium[box2d] gymnasium[other]``` 

3. Install torch with CUDA (e.g CUDA 11.8): 

   ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
   
# 🧛‍♂️ Create ".venv_tensorflow"
You need correct CUDA, CUDA Toolkit and cudNN versions installed according to your GPU.

* Python 3.10 version
* Tensorflow 2.9 version 

# 📂 Folder "results"
Results of both subprojects, including:

* Reward histories and test videos from the Lunar Lander.
* Test images colorization from the GAN.

# ⚖️ License
From Lunar Lander to colonizing Mars.

From GAN colorization to paint better than Picasso.

# Authors

* Alejandro Mendoza [pintamonas4575](https://github.com/pintamonas4575)


