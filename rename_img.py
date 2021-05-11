import glob
import os

def rename_img(path, add_name):
    for img in glob.glob(path+"/*"):
        base_name = os.path.basename(img)
        os.rename(img, path+"/"+add_name+"_"+base_name)

if __name__ == '__main__':
    list_name = ["id", "phone", "vr_number"]
    path = './data/file_rename/'
    for i in range(len(list_name)):
        rename_img(path+list_name[i],list_name[i])
        print(f"Rename {list_name[i]} is done")
    print("RENAME IS DONE")

