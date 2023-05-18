import models
import cv2
import torch
import utils
from torchvision import transforms
from torchvision import transforms

def load_model(path, model):
    model.load_state_dict(torch.load(path))
    return model

def predict(img):
    model = models.UNet(n_channels=3, n_classes=1)
    model = load_model('model.pt',model)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = cv2.resize(img, (512, 512))
    convert_tensor = transforms.ToTensor()
    img =  convert_tensor(img).float()
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)
    output = model(img)
    result = torch.sigmoid(output)
    threshold = 0.5
    result = (result >= threshold).float()
    prediction = result[0].cpu()  # Move tensor to CPU if it's on GPU
    # Convert tensor to a numpy array
    prediction_array = prediction.numpy()
    # Rescale values to the range [0, 255]
    prediction_array = (prediction_array * 255).astype('uint8').transpose(1, 2, 0)
    cv2.imwrite("test.png",prediction_array) 
    return prediction_array
