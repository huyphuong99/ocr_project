import glob
import os

import cv2
import pandas as pd

import xml.etree.ElementTree as ET

df_label = pd.read_csv("/media/data_it/Data_set/database_image/card/contract/info/cus_info.csv", dtype={
    'Phone': str,
    'NationalCard': str,
    'BankAccountNumber': str,
    'LoanAmount': int,
    'PlateNumber': str
})


def get_label(loan_id):
    row = df_label.loc[df_label['LoanBriefId'] == loan_id]
    data = row.to_dict(orient='records')
    if len(data) > 0:
        return data[0]
    return None


def get_label_box(annotation):
    result = {}
    for box in annotation['object']:
        label = box['name']
        if label in result:
            result[label].append(box['box'])
        else:
            result[label] = [box['box']]

    for k, v in result.items():
        v = sorted(v, key=lambda x: x[0])
        result[k] = v
    return result


filenames = []
label_texts = []
types = []


def crop_data(filename_image, image, label, label_box, output_dir):
    for k, v in label_box.items():
        sub_output_dir = os.path.join(output_dir, k)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        if k != 'phone':
            continue
        if k == 'fullname':
            label_text = label['FullName']
            label_text = label_text.split(' ')
        elif k == 'issue_date':
            label_text = ['unknown']
        elif k == 'id':
            label_text = [label['NationalCard']]
        elif k == 'birthday':
            label_text = [label['DOB']]
        elif k == 'bank':
            label_text = [label['BankCode']]
        elif k == 'bank_number':
            label_text = [label['BankAccountNumber']]
        elif k == 'money':
            label_text = [label['LoanAmount']]
        elif k == 'signature':
            label_text = [label['FullName']]
        elif k == 'vr_plate':
            label_text = [label['PlateNumber']]
        elif k == 'vr_name':
            label_text = label['FullName']
            label_text = label_text.split(' ')
        elif k == 'phone':
            label_text = [label['Phone']]
        elif k == 'text':
            print("text:", filename_image)
            continue

        else:
            continue
        for i, box in enumerate(v):
            filename = f'{filename_image}_{i}.jpg'
            output_path = os.path.join(sub_output_dir, filename)
            obj = image[box[1]: box[3], box[0]: box[2]]
            cv2.imwrite(output_path, obj)
            filenames.append(filename)
            try:
                label_texts.append(label_text[i])
            except IndexError:
                label_texts.append("")
                print('Index error:', filename_image)
        types.extend([k] * len(v))


def run(input_dir_image, output_dir):
    input_dir = "{}/*.xml".format(input_dir_image)
    for label_path in glob.glob(input_dir):

        tree = ET.parse(label_path)
        root = tree.getroot()

        annotation = {
            'folder': root.find('folder').text,
            'filename': root.find('filename').text,
            'path': root.find('path').text,
            'size': {
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'depth': int(root.find('size/depth').text),
            },
            'segmented': root.find('segmented').text,
            'object': []
        }
        filename, ext = os.path.splitext(annotation['filename'])
        # if os.path.exists(f"{output_dir}/fullname/{filename}_0.jpg"):
        #     continue
        image_path = os.path.join(input_dir_image, annotation['filename'])
        image = cv2.imread(image_path)
        if image is None:
            print("Not exists: ", image_path)
            continue
        idx, file_id, loan_id = filename.split('_')
        label = get_label(int(loan_id))
        for boxes in root.iter('object'):
            y1 = int(boxes.find("bndbox/ymin").text)
            x1 = int(boxes.find("bndbox/xmin").text)
            y2 = int(boxes.find("bndbox/ymax").text)
            x2 = int(boxes.find("bndbox/xmax").text)
            box = {
                'name': boxes.find('name').text,
                'pose': boxes.find('pose').text,
                'truncated': boxes.find('truncated').text,
                'difficult': boxes.find('difficult').text,
                'box': [x1, y1, x2, y2]
            }
            annotation['object'].append(box)

        label_box = get_label_box(annotation)
        crop_data(filename, image, label, label_box, output_dir)

    df = pd.DataFrame({'type': types, 'filename': filenames, 'label': label_texts})
    output_label_text = os.path.join(output_dir, 'label_text.csv')
    df.to_csv(output_label_text, index=False)


if __name__ == '__main__':
    run("/media/data_it/Data_set/database_image/card/contract/cropped/ttsdtb",
        "/media/data_it/Data_set/database_image/card/contract/info_ttsdtb")