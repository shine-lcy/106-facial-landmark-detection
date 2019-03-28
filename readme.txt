数据预处理gen_img.py：
将标注的106点数据存储为.txt形式，按照106个点的顺序，先存某点的x坐标，再存其y坐标。 其中一张图对应一个txt文件，其命名相同。
运行gen_img.py，其中需要自定义图片存储路径imgpath、txt文件存储路径landmark、新的txt文件存储save_landmark_anno以及数据增广之后的图片存储路径landmark_imgs_save_dir

训练train.py：
将已经预处理完成的新的txt文件save_landmark_anno传递给train.py文件中的label_path参数，运行train.py即可开始训练，模型文件存储在当前目录下的"my_net.pkl"（可自定义位置）。

测试test.py：
将所需要测试的图片路径传递给train.py中img_path变量，运行test.py即可测试。
