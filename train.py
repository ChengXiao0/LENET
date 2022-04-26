import numpy as np

from model.lenet import leNet
from dataset.mydata import mydata

from torch.utils.data import Dataset

import torch

device = torch.device("cuda:0")
batch = 16
epochs = 40

net = leNet().to(device=device)

dat = mydata("/home/xiaoxiao/PycharmProjects/imgResize/dataset/data")
dats = torch.utils.data.DataLoader(dat, batch_size=batch, shuffle=True, num_workers=2)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss = torch.nn.CrossEntropyLoss()
loss = loss.to(device)

for epoch in range(epochs):
    print("-----------------------------------------")
    tR = 0
    for batch_idx, (label, img) in enumerate(dats):
        img = img.float()
        label = label.to(device)
        img = img.to(device=device)
        opt.zero_grad()
        outLabelList = net(img)
        lossData = loss(outLabelList, label - 1)
        lossData.backward()
        opt.step()
        # print(lossData.data)
        checkA = outLabelList.data.cpu().numpy()
        checkB = label.data.cpu().numpy()
        resA = np.argmax(checkA, 1)
        for i in range(len(resA)):
            if resA[i] == checkB[i] - 1:
                # print("-y-")
                tR += 1
            else:
                pass
                # print("-n-")
    print(tR)
    acc = tR / 487
    print("acc = ", tR / 487)
    if acc > 0.85:
        torch.save(net.state_dict(), f'/home/xiaoxiao/PycharmProjects/imgResize/ckp/model{epoch}_{acc}.pth')
