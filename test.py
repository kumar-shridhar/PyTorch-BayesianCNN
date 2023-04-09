import torch
import cv2
import src.srcnn as srcnn
import numpy as np
import glob as glob
import os
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = srcnn.Net(1).to(device)
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}
model = srcnn.BayesianNet(1, priors).to(device)
model.load_state_dict(torch.load('./outputs/model.pth'))


image_paths = glob.glob('./input/bicubic_2x/*')
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    test_image_name = image_path.split(os.path.sep)[-1].split('.')[0]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    cv2.imwrite(f"./outputs/test_{test_image_name}.png", image)
    image = image / 255. # normalize the pixel values
    cv2.imshow('Greyscale image', image)
    cv2.waitKey(0)


    model.eval()
    with torch.no_grad():
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        outputs = model(image)

    outputs = outputs.cpu()
    save_image(outputs, f"./outputs/output_{test_image_name}.png")
    outputs = outputs.detach().numpy()
    outputs = outputs.reshape(outputs.shape[2], outputs.shape[3], outputs.shape[1])
    print(outputs.shape)
    cv2.imshow('Output', outputs)
    cv2.waitKey(0)