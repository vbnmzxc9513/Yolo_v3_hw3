import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from os import listdir
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


#X1 Y1 X2 Y2
test_img = listdir('./test/')
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_num.pb"
image_path      = "./train/5.png"
num_classes     = 11
input_size      = 416
graph           = tf.Graph()
output_list     = []

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

for i in range(len(test_img)):
    original_image = cv2.imread('./test/' + test_img[i])
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]


    with tf.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)


    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    print(i)
    output = {'bbox':[], 'label':[], 'score':[]}
    for j in range(len(bboxes)):
        output['bbox'].append([int(round(bboxes[j][1])), int(round(bboxes[j][0])), int(round(bboxes[j][3])), int(round(bboxes[j][2]))])
        output['label'].append(int(bboxes[j][5]))
        output['score'].append(bboxes[j][4])
    output_list.append(output)

with open('0857214.json', 'w') as outfile:
    json.dump(output_list, outfile)
    #image = utils.draw_bbox(original_image, bboxes)
    #image.savefig('test.png')
    #cv2.imwrite('test.png', image)
    #image.show()




