# 🧞‍♂️🎅🦹‍♂️RLGAN-project-MAADM-UPM 👽🤖👹
Neuroevolution to learn the Lunar Lander from Gymnasium and a GAN to learn to color images. Subject from the ML and BD master´s degree of UPM.

# 🧙‍♂️ Create ".venv_torch"
You need correct CUDA, CUDA Toolkit and cudNN versions installed according to your GPU. Used here:

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
You need correct CUDA, CUDA Toolkit and cudNN versions installed according to your GPU. Used here (from an Anaconda environment):

* Python 3.10 version
* Tensorflow 2.9 version 

# 📓 Notebook *RLGAN-p1-AG.ipynb*
Notebook for the first practical task, the lunar lander with reinforcement learning using a genetic algorithm.
The chromosomes are a MLP (class in file *MLP.py*) total weights and biases converted to lists. The Genetic Algorithm
process is in the file *AGLunarLander.py*.

# 📓 Notebook *RLGAN-p1-MLP.ipynb*
Notebook as a variant of the previous solution using a MLP in Pytorch, optimizing the process with NVIDIA GPU.

# 📓 Notebook *RLGAN-p2-GAN.ipynb*
Notebook where a GAN is created and trained in Tensorflow to color images from CIFAR-10 dataset. 

# 📂 Folder "results_p1"
Results of both solution types, including:

* 📈 Reward histories and test videos from the Lunar Lander.
* 🤖 *lunar_lander_AG.txt* file: Model from the GA version with the best global chromosome.
* 🤖 *lunar_lander_MLP.pt* file: Model from the MLP version with the weights and optimizer states.

# 📂 Folder "results_p2"
Folder where all the created subfiles for the GAN task are stored (ignored because too much and very heavy files). It mainly contains:

* Test images colorization plot from the GAN.

# ⚖️ License
From Lunar Lander to colonizing Mars (SpaceX hire us).

From GAN colorization to paint better than Picasso (We wish).

# 👥 Authors

* Alejandro Mendoza [@pintamonas4575](https://github.com/pintamonas4575)
* Jaime Álvarez     [@JaimeAlvarez434](https://github.com/JaimeAlvarez434)
* Álvaro Fraile     [@alvarofraile](https://github.com/alvarofraile)


