1. There are two folders along with this document. 
	a. Ubuntu 64.zip: it contains the virtual image with code, data and all dependencies installed.
	b. Cells: it contains all training and testing code along with some annotations and images. All data cannot be shared due to IP issues.

In the cells directory, there are many sub-folders. A brief description of each folder is as follows:
	a. 'annotation' contains xml files containing annotation for corresponding image.
	b. 'images' contains the raw images (gif format).
	c. 'notebooks' contains a self-contained jupyter notebook that shows how to use the trained model to detect cells in an image.
	d. 'scripts' folder contains python scripts for data preparation.
	e. 'src' folder contains source code for training the proposed model.
	f. 'utils' contains necessary python scripts for training and data preparation.
	g. 'weights' contains weights of trained models.

'patches' and 'png_images' will be created during training/testing of the model.


2. Installation:

a. If you want to use virtual image, please download 'Ubuntu 64.zip' and extract. Then load it into vmplayer. You can download and install free vmplayer from here: https://my.vmware.com/en/web/vmware/free#desktop_end_user_computing/vmware_workstation_player/14_0

In order to load the virtual image in vmplayer, go to file->Open a Virtual machine and then select Ubuntu 64-bit.vmx. Then click on power on and log in using the password 'cell1234' (without quotations). You will see a cells folder in the desktop. Open a terminal inside the folder and follow the instructions on Usage section.

b. If you are using linux, you can also download the cells folder containing data and source code. Please install the following python dependencies first.

	pip install scikit-learn scikit-image pillow numpy keras progressbar2 tqdm opencv-python pandas jupyter notebook ipython

If you have GPU, please follow the next instruction.
	
	pip install tensorflow-gpu

Otherwise,
	
	pip install tensorflow



3. Usage:
	
	At first, setup the directory to cells folder. You need to do this in a terminal. To open a terminal, double-click to open cells folder. Inside the folder, right-click on the white space (not on an folder), then select open a terminal. Then execute the following command.

		cellsdir=`pwd`

	If you have downloaded the cells folder, please execute the following commands:

		cd $cellsdir/scripts
		python data_preparation.py
		python gif_to_png.py

	In order to train and test to reproduce our result, execute the following commands: (this is not recommeded since we could not provide all data)

		cd $cellsdir/src
		python densenet_scratch.py


	In order to only test and reproduce the result in the paper, execute

		python densenet_scratch.py --reproduce True



There is a fully working and self-contained notebook for processing microscopic images to classify cells is included in the notebooks folder.
