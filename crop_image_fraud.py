import cv2


# coordinate = [(x_max, y_max), (x_min, y_min)]
def crop_img(image, coordinate, ratio=1 / 15.0):
    x_max, y_max = coordinate[0]
    x_min, y_min = coordinate[1]
    w_box = x_max - x_min
    h_box = y_max - y_min
    r = w_box / h_box
    x_min_new = x_min - int(ratio * w_box)
    y_min_new = y_min - int(r * ratio * h_box)
    x_max_new = x_max + int(ratio * w_box)
    y_max_new = y_max + int(r * ratio * h_box)
    img = image[y_min_new: y_max_new, x_min_new: x_max_new]
    return img


# coordinates
# [[(x_max, y_max), (x_min, y_min)],[]...]
def read_img(path: str, coordinates: list[tuple]):
    raw_img = cv2.imread(path)
    boxes = sorted(coordinates, key=lambda x: x[1])
    label = {0: "id", 1: "name", 2: "birthday"}
    dict_label = {}
    for i, box in enumerate(boxes):
        dict_label[i] = crop_img(raw_img, box)
        cv2.imshow(f"{i}", dict_label[i])
        cv2.waitKey()
    return dict_label


if __name__ == "__main__":
    path = "/home/huyphuong99/Desktop/material/test/id/test_1.jpg"
    coor = [[(438, 134), (306, 116)], [(453, 232), (359, 216)], [(491, 169), (297, 151)]]
    result = read_img(path, coor)
