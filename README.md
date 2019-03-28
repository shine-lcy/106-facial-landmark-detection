# 106-facial-landmark-detection
## Installation 
python 2.7  
pytorch 0.4.1  
opencv-python  
dlib python  
## Usage
### data preprocess
The marked 106-point data should be stored in the .txt format. According to the order of 106 points, the x coordinate of a certain point is stored first, and then the y coordinate is stored. One of the maps corresponds to a txt file with the same name.

`python gen_img.py`

It requires a origin image storage path **imgpath**, a txt file storage path **landmark**, a new txt file storage **save_landmark_anno**, and an image storage path after the data augmentation **landmark_imgs_save_dir**.

### train
Assign **save_landmark_anno** to **label_path** in **train.py**.

`python train.py`

The model file is stored as **my_net.pkl** in the current directory.

### test
Assign the test image file path to **img_path** in **test.py**.

`python test.py`

Then drink a cup of coffee and start enjoying the joy of success.
