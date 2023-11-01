import time
import argparse
from multiprocessing.pool import ThreadPool
import multiprocessing




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
file_list = list(os.walk(path))
img_list = []

def file_checker(idx):
    r, d, f = file_list[idx]
    for file in f:
        for fileformat in FILEFORMATS:    
            if fileformat in file:
                try:
                    img = Image.open(os.path.join(r, file))
                    #img.verify()
                    width_div = img.width // args.max_size
                    height_div = img.height // args.max_size
                    div = min(width_div, height_div)
                    if div > 1:
                        img_resize = img.resize((int(img.width/div),int(img.height/div)),Image.BILINEAR)
                        img.close()   
                        img_resize.save(os.path.join(r,file))
                        #print("Image too large:", os.path.join(r,file))             
                    break
                    # print("DONE", os.path.join(r, file))
                except:
                    img_list.append(os.path.join(r, file) + "\n")
                    print("Something wrong with image:", os.path.join(r, file))
                    break
    if idx % 10000 == 0:
        print(f'idx {idx} fin.')
        
with ThreadPool(multiprocessing.cpu_count()) as pool1:
    pool1.map(file_checker, range(len(file_list)))
    pool1.close()
    pool1.join()
    
print("time :", time.time() - start)  

for img in img_list:
    os.remove(img)
    print("Image removed: ", img)