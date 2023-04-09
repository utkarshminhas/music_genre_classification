# what finally worked
## before getting to the video link, a few pointers
	- all we need to do is install the nvidia cuda toolkit and the cudnn; which is already there in the video
	- no need for any external installation apart from the stuff i am mentioning here
## which versions to select for installation
- as for the VERSION of the softwares needed to be installed
	- CHECK THE MINIMUM CUDA TOOLKIT VERSION WHICH YOUR GPU SUPPORTS AND ANY VERSION FROM THAT AND ABOVE IS THE ELIGIBLE VERSION FOR ME TO INSTALL IN MY MACHINE
	- Then, check these 2 places
	- https://www.tensorflow.org/install/source
		- check the CUDA VERSION and the CuDNN versions here for the latest versions available for TF
	- https://pytorch.org/
		- check the CUDA VERSIONS here for the latest versions of pytorch
	- Since I would be using these two in my machine for ML,DL etc. I need to decide for one version of CUDA which can work for both tensorflow and torch. the cuda version for both of them needs to be the same, tf version and pytorch version doesnt matter here.
## steps for installation
- install gcc
- upgrade the c++11 to c++14 or higher(the older version gives a warning during cuda-samples testing, but it will get installed anyways. This is an optional step)
- start with this video
	- Video name - How To Install CUDA, cuDNN, Ubuntu, Miniconda | ML Software Stack | Part 3/3
	- Channel name - "Aleksa Gordić - The AI Epiphany"
	- https://youtu.be/ttxtV966jyQ
	- start from the "installing cuda" part
- while installing the cuda toolkit, you might get a prompt to enter "OK" on a Purple screen with grey box, prompting you to enter a password.
	- Do it and remeber that password 
	- reboot the machine
	- press "enroll"
	- this should take you to the screen which gives you 3 options, choose the first one. The other options tell you to enroll a key from external sources like disk. we dont need to do that because the installation process from the video already generated the key for us and we need to enroll that key. So, choose the first option and move on
	-  There should be 2 options on the screen now
		- View key 0
		- Continue
	- Select "Continue", enter the password to enroll the key, and resume booting the machine to finish this process and move on the next steps in the video
	- We did this so that we can enroll the generated key which we generated from performing the steps in the video
- you might get an error while performing the test on the "cuda-samples" repo which belongs to this blog
	- https://forums.developer.nvidia.com/t/freeimage-is-not-set-up-correctly-please-ensure-freeimae-is-set-up-correctly/66950
	- Error name : FreeImage is not set up correctly. Please ensure FreeImae is set up correctly 
	- Implement the following commands to fix these things
	- `sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev`
	- `sudo apt-get install libfreeimage3 libfreeimage-dev`



# copy commands from these links while referring to the video
- Nvidia CUDA installation docs
	- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#pre-installation-actions
- The NVIDIA CUDA Toolkit is available at :
	- https://developer.nvidia.com/cuda-downloads
	- Select the right version(any)
		- refer https://www.tensorflow.org/install/source for which version of CUDA to install, refer https://pytorch.org/ 
		- Select a CUDA version which works with both of them, and remember to install tf/pyotrch while keeping those versions in mind during your development
- CuDNN link
	- https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
	- refer https://www.tensorflow.org/install/source for which version of CuDNN to install based on which CUDA driver you have already installed


# THE stuff below is just for reference, and not a part of the installation process

## installation time the first time around...... when it worked.....somewhat... and never worked again after that. 
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
- https://github.com/nvidia/cuda-samples
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local




### need to try the conda approach????
- https://www.tensorflow.org/install/pip
