import glob
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict

from lxml import etree
import traceback


def sort_box(boxes: List[Dict]) -> List[List[Dict]]:
    list_boxes = boxes.copy()
    results: List[List[Dict]] = []
    append_factor = 0.3
    if len(list_boxes) > 1:
        vertical_sorted = sorted(list_boxes, key=lambda b: b['box'][1])
        line = []
        y_center_line = (vertical_sorted[0]['box'][1] + vertical_sorted[0]['box'][3]) / 2
        min_range = y_center_line - (
                vertical_sorted[0]['box'][3] - vertical_sorted[0]['box'][1]) * append_factor
        max_range = y_center_line + (
                vertical_sorted[0]['box'][3] - vertical_sorted[0]['box'][1]) * append_factor
        for box in vertical_sorted:
            if min_range < (box['box'][1] + box['box'][3]) * 0.5 < max_range:
                line.append(box)
            else:
                results.append(sorted(line, key=lambda x: x['box'][0]))
                line = [box]
                y_center_line = (box['box'][1] + box['box'][3]) / 2
                min_range = y_center_line - (box['box'][3] - box['box'][1]) * append_factor
                max_range = y_center_line + (box['box'][3] - box['box'][1]) * append_factor
        results.append(sorted(line, key=lambda x: x['box'][0]))
        len(results)
    return results


def update_line_label(line, label):
    for box in line:
        box['name'] = label


def iterator_lines(lines: List[List[Dict]], image_width, image_height):
    key_process = 'text'

    conditionals = [
        {
            'position': -1,
            'name': 'vr_name'
        },
        {
            'position': -2,
            'name': 'vr_number'
        },
        {
            'position': -3,
            'name': 'phone'
        },
        {
            'position': -4,
            'name': 'id'
        },
        {
            'position': 0,
            'name': 'fullname'
        }
    ]

    is_set = {}
    for cond in conditionals:
        is_set[cond['name']] = False

    for cond in conditionals:
        category = cond['name']
        position = cond['position']
        if not is_set[category]:
            try:
                line = lines[position]
            except IndexError as e:
                print(e, category, position)
                break
            if line[0]['name'] == key_process:
                update_line_label(line, category)
                print(f'\t\tLine {position} change label to "{line[0]["name"]}')
            else:
                print(f'\t\tLine {position} had label by "{line[0]["name"]}"')
            is_set[category] = True
    return lines


def process_annotation(annotation):
    lines = sort_box(annotation['object'])
    lines = iterator_lines(lines, annotation['size']['width'], annotation['size']['height'])
    annotation['object'] = []
    for line in lines:
        for box in line:
            annotation['object'].append(box)
    return annotation


def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def write_annotation(annotation_data: Dict, output_path: str):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = annotation_data['folder']
    ET.SubElement(annotation, 'filename').text = annotation_data['filename']
    ET.SubElement(annotation, 'path').text = annotation_data['path']
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(annotation_data['size']['width'])
    ET.SubElement(size, 'height').text = str(annotation_data['size']['height'])
    ET.SubElement(size, 'depth').text = str(annotation_data['size']['depth'])
    ET.SubElement(annotation, 'segmented').text = '0'

    for box in annotation_data['object']:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = box['name']
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(box['box'][0])
        ET.SubElement(bbox, 'ymin').text = str(box['box'][1])
        ET.SubElement(bbox, 'xmax').text = str(box['box'][2])
        ET.SubElement(bbox, 'ymax').text = str(box['box'][3])

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, output_path)


def asign_cl(annotation):
    temp = 0
    for (idx, box) in enumerate(annotation['object']):
        if box['name'] == 'vr_number':
            temp = idx
            break
    try:
        if annotation['object'][temp]['box'][0] > annotation['object'][temp + 1]['box'][0]:
            annotation['object'][temp]['name'] = 'vr_plate'
        else:
            annotation['object'][temp + 1]['name'] = 'vr_plate'
    except Exception as e:
        annotation['object'][temp]['name'] = 'vr_number'
        print(e)
    return annotation


def run(input_dir):
    input_dir = "{}*.xml".format(input_dir)
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
        print(f'Process file: {annotation["filename"]}')
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
        annotation = process_annotation(annotation)
        annotation = asign_cl(annotation)
        write_annotation(annotation, label_path)


if __name__ == '__main__':
    input_dir = "./data/ttsdtb_xml/"
    run(input_dir)
