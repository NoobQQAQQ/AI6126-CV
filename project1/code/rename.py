import os

if __name__ == '__main__':
    path = "../data/addition/val_mask/"
    file_list = os.listdir(path)
    for i in file_list:
        oldname = path + i
        newname = path + 'v' + i
        os.rename(oldname, newname)