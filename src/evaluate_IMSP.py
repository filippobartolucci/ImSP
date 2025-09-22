import os
import utils
import datetime
from tqdm import tqdm

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from Models.MaLP_Models import DetectionModule
from Models.SpatialTransformer import SpatialTransformer, SignalRecovery
from Generators.StableDiffusionImg2Img import StableDiffusionImg2Img


test_dir = 'PATH_TO_DATASET'
model_dir = 'PATH_TO_MODELS'
prompt = "PROMPT_FOR_MANIPULATION"


utils.fix_random(42)
device = utils.get_device()
print("Device: ", device)


class IMG_Dataset(Dataset):
    def __init__(self, data_dir: str, img_size:128):
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_list = os.listdir(data_dir)
 
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


# Dataset
test_dataset = IMG_Dataset(test_dir, 128)


# Models
encryption_module = SpatialTransformer(depth=1).to(device).eval()
encryption_module.load_state_dict(torch.load(os.path.join(model_dir, 'signal_encoder.pth')))
decryption_module = SignalRecovery(depth=1).to(device).eval()
decryption_module.load_state_dict(torch.load(os.path.join(model_dir, 'signal_decoder.pth')))
detector = DetectionModule().to(device).eval()
detector.load_state_dict(torch.load(os.path.join(model_dir, 'class_model.pth')))


GM = StableDiffusionImg2Img()


# Test set
batch_size = 2
test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Test size: ", len(test)*batch_size)


# Detection Accuracy
detection_correct = 0
total = 0


print("Evaluation...", end="\r")
for batch in tqdm(test):
    imgs= batch
    imgs = imgs.to(device)

    with torch.no_grad():
        protection = encryption_module(imgs)
        protected_img = imgs + protection
        protected_img = torch.nn.functional.interpolate(protected_img, size=(512,512), mode='bilinear', align_corners=False)

        # Manipulation happens here
        manipulated_img = GM.manipulate(protected_img, prompt)
                           
        protected_img = torch.nn.functional.interpolate(protected_img, size=(128,128), mode='bilinear', align_corners=False)
        manipulated_img = torch.nn.functional.interpolate(manipulated_img, size=(128,128), mode='bilinear', align_corners=False)

        # Signal Recovery
        signal_real = decryption_module(protected_img)
        signal_fake = decryption_module(manipulated_img)
        signal_real_N= utils.norm(signal_real.contiguous())
        signal_fake_N = utils.norm(signal_fake.contiguous())

        # Manipulation Detection
        comb_maps = torch.cat((signal_real, signal_fake), dim=0)
        gt_class=torch.zeros((imgs.shape[0]*2), dtype=torch.float, device=device)
        gt_class[int(comb_maps.shape[0]/2):]= torch.tensor([1], dtype=torch.float, device=device)
        gt_class = gt_class.unsqueeze(1)

        # Detection Accuracy
        pred_class = detector(comb_maps)
        pred_class = torch.sigmoid(pred_class)
        pred_class = torch.round(pred_class)
            
        detection_correct += torch.sum(pred_class == gt_class).cpu().detach().numpy()
        total += gt_class.size(0)

detection_acc_mean = round(detection_correct / total, 6)
print("Evaluation completed!")
print("\nDetection Accuracy: ", detection_acc_mean)
