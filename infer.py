import numpy as np
import cv2
import torch
import torchvision.transforms as tr
from model.lenet import leNet


device = torch.device('cpu')
checkpoint = torch.load("ckp/model67_0.8624229979466119.pth")
model = leNet()
model.load_state_dict(checkpoint)
model.eval()
model.to(device)
torch.set_grad_enabled(False)


video = cv2.VideoCapture(0)
while video.isOpened():
    ret, frame = video.read()
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', frame)
    frame = torch.tensor(frame)
    frame = torch.unsqueeze(frame, 0)
    frame = torch.unsqueeze(frame, 0)
    frame = frame.float()
    out = model(frame)
    checkout = out.data.numpy()
    label = np.argmax(checkout)
    print(label+1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
