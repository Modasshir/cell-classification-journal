### 1. Directory information
In the cells directory, there are many sub-folders. A brief description of each folder is as follows:
	<ol>
	<li>'annotation' contains xml files containing annotation for corresponding image.</li>
	<li>'images' contains the raw images (gif format).</li>
	<li>'notebooks' contains a self-contained jupyter notebook that shows how to use the trained model to detect cells in an image.</li>
	<li>'scripts' folder contains python scripts for data preparation.</li>
	<li>'src' folder contains source code for training the proposed model.</li>
	<li>'utils' contains necessary python scripts for training and data preparation.</li>
	</ol>
	
Please create a folder in the root directory of the project named 'weights'. Then download <a href='https://drive.google.com/file/d/1h9-Xo12b2QoTaGlGdAP6tFyhj1ckLrKH/view?usp=sharing'>reproducible weights</a> and put this inside 'weights' folder.

'patches' and 'png_images' will be created during training/testing of the model.


### 2. Installation:

The installation instructions are for linux/ubuntu. For Windows, the approach is similar, however, not tested. If you are using Anaconda(recommended), please replace the following 'pip' in the following instructions with 'conda'. 

	pip install scikit-learn scikit-image pillow numpy keras progressbar2 tqdm opencv-python pandas jupyter notebook ipython

If you have GPU, please follow the next instruction.
	
	pip install tensorflow-gpu

Otherwise,
	
	pip install tensorflow



### 3. Usage:
	
Open a terminal and go to the project directory. Then execute the following command.

	cellsdir=`pwd`

If you have downloaded the cells folder, please execute the following commands:

	cd $cellsdir/scripts
	python data_preparation.py
	python gif_to_png.py

In order to train and test, execute the following commands: (this is not recommeded since we could not provide all data due to confidentiality issues.)

	cd $cellsdir/src
	python densenet_scratch.py


In order to only test and reproduce the result in the paper, execute

	python densenet_scratch.py --reproduce True


There is a fully working and self-contained notebook for processing microscopic images to classify cells is included in the notebooks folder.

### 4. License

This project was developed at the University of North Carolina at Chapel Hill.
The open-source version is licensed under the GNU General Public License
Version 3 (GPLv3).
