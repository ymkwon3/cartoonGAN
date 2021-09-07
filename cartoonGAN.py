import torch
import torchvision.transforms as transforms
import cv2
import os
import matplotlib.pyplot as plt

from network.Transformer import Transformer

model = Transformer()
model.load_state_dict(torch.load('pretrained_model/Hayao_net_G_float.pth'))
model.eval()
print('Model loaded!')

img_size = 700
img_path = 'test_img/9.jpg'

img = cv2.imread(img_path)

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size, 2),
    transforms.ToTensor()
])

img_input = T(img).unsqueeze(0)
img_input = -1 + 2 * img_input

img_output = model(img_input)
img_output = (img_output.squeeze().detach().numpy() + 1.) / 2.
img_output = img_output.transpose([1,2,0])

cv2.imshow("img", img_output)
img_output = cv2.convertScaleAbs(img_output, alpha=(255.0))
cv2.imwrite('test_output/img.jpg', img_output)

cv2.waitKey(0)