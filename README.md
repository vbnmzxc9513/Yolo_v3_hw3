# Yolo_v3_hw3
Heavily Reference from: https://github.com/YunYang1994/tensorflow-yolov3

Training Process
1.mat2txt.m: change .mat to .txt to build annotation. 
The format is path image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 

2./core/config: Set config in core.  __C.TRAIN.ANNOT_PATH = "./data/dataset/num_train.txt"

3.train.py: This step will take a long time. You will see checkpoint in the checkpoint folder.

4.freeze_graph.py: Change the .ckpt file to .pb file

5.image_demo.py: Demo the image. Box the target in the image.

6.Jason.py: To generate the .json file contain the information of test folder.

7.speed_benchmark.ipynb: Get the speed score from co-lab.

# Result
![image](https://github.com/vbnmzxc9513/Yolo_v3_hw3/blob/master/test2.png)

# Speed (Colab GPU)
![image](https://github.com/vbnmzxc9513/Yolo_v3_hw3/blob/master/speed_benchmark.bmp)

very slow...
