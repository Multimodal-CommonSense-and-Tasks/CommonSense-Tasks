import time
import argparse

parser = argparse.ArgumentParser("Dataset cleaning")
parser.add_argument("-p", "--path", type=str, default=None, help="directory to inspect")
parser.add_argument("-s", "--max_size", type=int, default=1024, help="image max size")
args = parser.parse_args()

FILEFORMATS = ['jpg', 'jpeg', 'png', 'bmp']

start = time.time()  # 시작 시간 저장

from PIL import Image
import os

path = args.path

print(path)

check_img_list = os.path.join(path, "check_imgs.txt")
check_txt_list = os.path.join(path, "check_txts.txt")
check_number = os.path.join(path, "check_number.txt")
img_list = open(check_img_list, "w")
txt_list = open(check_txt_list, "w")

count = 0
total = 0
count_text = 0
total_text = 0

for r, d, f in os.walk(path):
    for file in f:
        if file.endswith(".txt"):
            try:
                tmp = open(os.path.join(r, file), "r")
                tmp.close()
                total_text += 1
            except:
                txt_list.write(os.path.join(r, file) + "\n")
                count_text += 1
        for fileformat in FILEFORMATS:    
            if fileformat in file:
                try:
                    img = Image.open(os.path.join(r, file))
                    #img.verify()
                    width_div = img.width // args.max_size
                    height_div = img.height // args.max_size
                    div = min(width_div, height_div)
                    if div > 1:
                        img_resize = img.resize((int(img.width/div),int(img.height/div)),Image.LANCZOS)
                        img.close()   
                        img_resize.save(os.path.join(r,file))
                        #print("Image too large:", os.path.join(r,file))             
                    total += 1
                    break
                    # print("DONE", os.path.join(r, file))
                except:
                    img_list.write(os.path.join(r, file) + "\n")
                    print("Something wrong with image:", os.path.join(r, file))
                    count += 1
                    break


with open(check_number, "w") as num:
    num.write("total num of imgs:" + str(total) + "\n")
    num.write("total num of corrupted imgs:" + str(count) + "\n")
    num.write("total num of txts:" + str(total_text) + "\n")
    num.write("total num of corrupted txts:" + str(count_text) + "\n")


img_list.close()
txt_list.close()
print(total, count)
print("time :", time.time() - start)  

remove_txt_list = os.path.join(path, "check_txts.txt")
with open(remove_txt_list, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        txt = line.strip()
        os.remove(img)
        print("Text removed: ", txt)

remove_img_list = os.path.join(path, "check_imgs.txt")
with open(remove_img_list, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        img = line.strip()
        os.remove(img)
        print("Image removed: ", img)