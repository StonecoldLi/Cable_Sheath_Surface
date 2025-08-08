import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version 4.7 MB

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    im = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    
    pb_np = np.array(imo)

    # Save the prediction output
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(os.path.join(d_dir, imidx + '.png'))


def main():
    # --------- 1. Set model and directories ---------
    model_name = 'u2net'  # your trained model name
    
    # Directories
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results')
    model_dir = os.path.join(os.getcwd(), 'saved_models', 'u2net', model_name + '.pth')

    # Get the list of test images
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(f"Test images found: {len(img_name_list)}")
    
    # --------- 2. DataLoader setup ---------
    # No labels needed for test (so leave lbl_name_list empty)
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), 
                                                                      ToTensorLab(flag=0)]))
    
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. Model Definition ---------
    print("Loading model...")

    # Choose the model (full version or smaller version)
    net = None
    if 'u2net' in model_name:
        print("Loading U2NET model...")
        net = U2NET(3, 1)
    elif 'u2netp' in model_name:
        print("Loading U2NETP model...")
        net = U2NETP(3, 1)
    else:
        print(f"Model name {model_name} is not recognized. Please check the model name.")
        return

    # Load the pre-trained model
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))  # Load model from GPU-trained file
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))  # Load model from CPU-trained file
        net.cpu()

    net.eval()  # Set the model to evaluation mode

    # --------- 4. Inference ---------
    print("Starting inference on test images...")

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print(f"Inferencing {img_name_list[i_test].split(os.sep)[-1]}")

        # Get the test image
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # Forward pass
        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # Use the first output (d1) as the prediction and normalize
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # Save the output to the prediction directory
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        # Clean up temporary variables to free memory
        del d1, d2, d3, d4, d5, d6, d7

    print("Inference complete!")


if __name__ == "__main__":
    main()
