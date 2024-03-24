import argparse
import torch
import random
import time
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision  import transforms
from captcha.image import ImageCaptcha
from tqdm import tqdm

captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size = 4

def texttovec(text):
    vectors=torch.zeros((captcha_size,captcha_array.__len__()))
    for i in range(len(text)):
        vectors[i,captcha_array.index(text[i])]=1
    return vectors

def vectotext(vec):
    vec=torch.argmax(vec,dim=1)
    text_label=""
    for v in vec:
        text_label+=captcha_array[v]
    return  text_label

class datasets(Dataset):
    def __init__(self,root_dir):
        super(datasets, self).__init__()
        self.list_image_path=[ os.path.join(root_dir,image_name) for image_name in os.listdir(root_dir)]
        self.transforms=transforms.Compose([
            transforms.Resize((60,160)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])
    def __getitem__(self, index):
        image_path = self.list_image_path[index]

        img_ = Image.open(image_path)
        img_tesor=self.transforms(img_)

        image_name=image_path.split("\\")[-1]
        img_lable=image_name.split("_")[0]

        img_lable=texttovec(img_lable)
        img_lable=img_lable.view(1,-1)[0]
        return img_tesor,img_lable
    def __len__(self):
        return self.list_image_path.__len__()

def create_datasets(total, test_ratio):
    image = ImageCaptcha()
    test_num = int(total * test_ratio)
    train_num = total - test_num
    with tqdm(total = total, desc = 'making progress') as pbar:
        for i in range(train_num):
            image_val = "".join(random.sample(captcha_array, 4))
            image_name = "./data/train/{}_{}.png".format(image_val, int(time.time()))
            image.write(image_val, image_name)
            pbar.update(1)
        for i in range(test_num):
            image_val = "".join(random.sample(captcha_array, 4))
            image_name = "./data/test/{}_{}.png".format(image_val, int(time.time()))
            image.write(image_val, image_name)
            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, required=True, help='Total number of images in the dataset')
    parser.add_argument('--test_ratio', type=float, required=True, help='The ratio of test set in the dataset')
    args = parser.parse_args()
    create_datasets(args.data_size, args.test_ratio)