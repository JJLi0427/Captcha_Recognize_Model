import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import *

def test_picture(pic_path, modelpath):
    img=Image.open(pic_path)
    plt.imshow(img)
    tersor_img=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60,160)),
        transforms.ToTensor()
    ])
    img=tersor_img(img).cuda()
    img=torch.reshape(img,(-1,1,60,160))
    m = torch.load(modelpath).cuda()
    outputs = m(img)
    outputs=outputs.view(-1,len(captcha_array))
    outputs_lable=vectotext(outputs)
    plt.title("model predict:{}".format(outputs_lable))
    plt.axis('off')
    plt.show()

def test_model(modelpath):
    m = torch.load(modelpath).cuda()
    m.eval()
    test_data = datasets("./data/test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()
    correct = 0
    with tqdm(total = test_length, desc = "Testing") as pbar:
        for i, (imgs, lables) in enumerate(test_dataloader):
            imgs = imgs.cuda()
            lables = lables.cuda()
            lables = lables.view(-1, captcha_array.__len__())
            lables_text = vectotext(lables)
            predict_outputs = m(imgs)
            predict_outputs = predict_outputs.view(-1, captcha_array.__len__())
            predict_labels = vectotext(predict_outputs)
            if predict_labels == lables_text:
                correct += 1
            pbar.update(1) 
    print("accuracy:{:.2%}".format(correct/test_length))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the model')
    parser.add_argument('--mode', type=str, required=True, choices=['test_model', 'test_picture'], help='Mode of operation')
    parser.add_argument('--picpath', type=str, help='Path to the picture (required if mode is test_picture)')
    args = parser.parse_args()

    if args.mode == 'test_model':
        test_model(args.modelpath)
    elif args.mode == 'test_picture':
        if args.picpath is None:
            raise ValueError('The --picpath argument is required when mode is test_picture')
        test_picture(args.picpath, args.modelpath)