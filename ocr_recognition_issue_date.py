import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #bo cac warming
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import pandas as pd
import cv2 as cv
import sys


def add_padding(image, img_w, img_h, filenames, list_file, shuffer=False):
    img = cv.imread(image)
    name = os.path.basename(image)
    hh, ww, cc = img.shape
    try:
        rate = ww / hh
        img = cv.resize(img, (round(rate * img_h), img_h), interpolation=cv.INTER_AREA)
        color = (0, 0, 0)
        result = np.full((img_h, img_w, cc), color, dtype=np.uint8)
        result[:img_h, :round(rate * img_h)] = img

    # return image, result
        if shuffer:
            cv.imwrite("./data/input/padding_test/" + name, result)
            list_file.append("./data/input/padding_test/" + name)
            filenames.add("./data/input/padding_test/" + name)
        else:
            cv.imwrite("./data/padding_test/" + name, result)
    except Exception as e:
        img = cv.resize(img, (img_w, img_h))
        if shuffer:
            cv.imwrite("./data/input/padding_test/" + name, img)
        else:
            cv.imwrite("./data/padding_test/" + name, img)



def prepare_data(NULL_CHAR, max_len, path):
    images = []
    labels = []
    dont_exit_file = []
    lb = []
    lb = []
    if os.path.exists(path + ".csv"):
        file_birth = pd.read_csv(path + ".csv", dtype={"label": str})
    for idx, row in file_birth.iterrows():
        image_path = os.path.join(path, row['filename'])
        if os.path.exists(image_path):
            images.append(image_path)
            label = row['label']
            lb.append(label)
            label_padded = [NULL_CHAR] * max_len
            label_padded[:len(label)] = label
            labels.append(label_padded)
        else:
            dont_exit_file.append(os.path.basename(image_path))

    return images, labels, lb


def encode_single_sample_test(img_path, name_img, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(label)
    return {"image": img, "basename": name_img, "label": label}


def load_data():
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, name_test, labels))
    test_dataset = test_dataset.map(encode_single_sample_test,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return test_dataset


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:,
              :max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def predict():
    predict = []
    basename = []
    for batch in test_dataset.take(5):
        batch_images = batch["image"]
        # batch_label = batch['label']
        batch_name = batch['basename']
        preds = model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        j = 0
        for i in range(len(pred_texts)):
            batch_n = str(batch_name[i].numpy(), "utf-8")
            basename.append(batch_n)
            j += i
            img = (batch_images[i, :, :, :] * 255).numpy().astype(np.uint8)
            img = img.transpose([1, 0, 2])
            label_predict = pred_texts[i].replace("[UNK]", "")
            # print("" + batch_n)
            # print(str(i) + " " + ": " + label_predict)
            # cv.imshow(str(batch_n), img)
            # cv.waitKey()
            predict.append(label_predict)
    # print(predict,"\n", len(predict))
    # print("basename: ", basename[0])
    # print("predict: ", predict)
    return predict, basename


def evaluated():
    # words
    count = 0
    for i in range(len(orig_texts)):
        if orig_texts[i] == predict[i]:
            count += 1
    acc = count / len(orig_texts) * 100
    print("Accuracy test data: " + str(round(acc, 2)) + "%")


def dist(x, m, y, n):
  if m == 0:
    return n
  if n == 0:
    return m
  cost = 0 if x[m - 1] == y[n - 1] else 1
  return min(dist(x, m - 1, y, n)+1, dist(x, m, y, n - 1) + 1, dist(x, m - 1, y, n - 1) + cost)



if __name__ == "__main__":
    path_test = "./data/input/"
    (batch_size, img_height, img_width, max_length) = (16, 40, 440, 12)
    NULL_CHAR = "<nul>"
    characters = "0123456789"
    characters = sorted(characters)
    characters.append(NULL_CHAR)
    characters.append(" ")

    images_1, labels_1, lb = prepare_data(NULL_CHAR, max_length, path_test + "id")
    # images_2, labels_2 = [], []
    # images_test = [f for f in glob.glob(path_test+"test/*")]
    images_test = images_1
    labels = labels_1
    filenames = set([])
    list_file = []
    for img in images_test:
        try:
            add_padding(img, img_width, img_height, filenames, list_file, shuffer=True)
        except Exception as e:
            pass
            print(e)
    # print(len(filenames))

    # for file in list(filenames):
    #     cnt =  list_file.count(file)
    #     if cnt > 1:
    #         print(f'File {file} existed {cnt} times')
    #
    #
    x_test = [file for file in glob.glob(path_test + "padding_test/*")]
    name_test = x_test

    char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters),
                                                                 num_oov_indices=1, mask_token='')
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), invert=True)

    test_dataset = load_data()
    print(test_dataset.take(1))
    # print(test_dataset)
    '''model = keras.models.load_model(
        "/home/huyphuong99/PycharmProjects/ocr_project/weight/number_model.h5")
    predict, base_name = predict()
    orig_texts = []
    images_t = [file[-27:] for file in images_test]
    # print(images_t)

    for path in base_name:
        idx = images_t.index(path[-27:])
        label = lb[idx]
        orig_texts.append(label)
    # print(predict)
    # print(orig_texts)
    evaluated()

    # print(predict, "\n", orig_texts)
    # for i in range(len(predict)):
    #     print("basename: {},predict: {}, original: {}".format(images_test[i],predict[i],orig_texts[i]))

    # evaluated()'''
