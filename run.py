import torch
import time
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *
from datasets import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workname', type=str, required=True, help='Name of the training work')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    args = parser.parse_args()

    work_dir = './checkpoint/' + args.workname
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    train_datas=datasets("./data/train")
    train_dataloader=DataLoader(train_datas,batch_size=160,shuffle=True)
    train_length = train_datas.__len__()
    test_data = datasets("./data/test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()
    m=model(captcha_size, captcha_array).cuda()
    lossfunc=nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    totalstep=0
    modelpath=work_dir + "/out.pth"
    epochnum=args.epochs

    loss_history = []
    accuracy_history = []
    start_time = time.time()
    for epoch in range(epochnum):
        with tqdm(total = int(train_length/160), desc = "train", ncols=150) as pbar:
            for i,(imgs,targets) in enumerate(train_dataloader):
                imgs=imgs.cuda()
                targets=targets.cuda()
                outputs=m(imgs)
                loss = lossfunc(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totalstep+=1
                pbar.update(1) 
                pbar.set_postfix({'loss':'%.6f' % (loss.item())})

                # if totalstep%100==0:
                #     print("times:{} loss:{:.6f}".format(totalstep, loss.item()))

        correct = 0
        with torch.no_grad():
            with tqdm(total = test_length, desc = "test ", ncols=150) as pbar:
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
                    pbar.set_postfix({'correct':'%d' % (correct)})
            accuracy = correct/test_length
            print("epoch:{} | loss:{:.6f} | test accuracy:{:.2%}\n".format(epoch+1, loss, accuracy))

        loss_history.append(loss.item())
        accuracy_history.append(accuracy)
        if epoch>0 and accuracy>accuracy_history[epoch-1]:
            torch.save(m, modelpath)
        if accuracy>0.99:
            break

    end_time = time.time()
    training_time = end_time - start_time
    if epoch+1==epochnum:
        print("\nend training | spend time: {:.3f} seconds".format(training_time))
        print("loss:{:.6f} accuracy:{:.2%}".format(loss.item(), accuracy))
    else:
        print("\nfinish training | spend time: {:.3f} seconds".format(training_time))
        print("epoch:{} loss:{:.6f} accuracy:{:.2%}".format(epoch+1, loss.item(), accuracy))
    print("model save to "+modelpath)

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.savefig(work_dir + '/loss.png')
    plt.figure()
    plt.plot(accuracy_history)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.savefig(work_dir + '/accuracy.png')
