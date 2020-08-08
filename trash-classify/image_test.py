# Predictive images category
import keras.backend as K 
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from models.model3 import M_b_Xception_896


K.clear_session()

input_shape = (229, 229, 3)
num_classes = 6
batch_size =   # The number of images extracted


testset_dir = 'data/test/'

weight_path = 'weights/ .h5/'
model = M_b_Xception_896(input_shape, num_classes)
model.load_weights(weight_path)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

test_x, test_y = test_generator.__getitem__(1)

preds = model.predict(test_x)

for i in range(batch_size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('img_test', cv2.WINDOW_NORMAL)
    pred_text = 'pred:%s' % (labels[np.argmax(preds[i])])
    truth_text = 'truth:%s' % (labels[np.argmax(test_y[i])])
    cv2.putText(test_x[i], pred_text, (7, 20), font, 0.3, (255, 0, 0), 1, cv2.LINE_4)
    cv2.putText(test_x[i], truth_text, (7, 50), font, 0.3, (0, 0, 255), 1, cv2.LINE_4)
    cv2.imshow('img_test', test_x[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()






