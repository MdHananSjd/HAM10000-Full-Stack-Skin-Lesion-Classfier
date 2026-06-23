import os
import pandas as pd
from src.utils.logger import logger
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HAM10000Dataset(Dataset):
    """custom dataset for handling ham10000 skin lesion image transofrms and hair removal"""
    def __init__(self, csv_file, img_dir, transform=None, clean_hair=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.clean_hair = clean_hair

        #label map for 7 disease categories
        self.label_map = {
            'actnh':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6
        }

        logger.info(f'Dataset initialized with {len(self.df)} samples. Hair removal set to: {clean_hair}')
    
    def __len__(self):
        return len(self.df)
    
    def __remove__hair__(self, image):
        """applies filter to remove hair artefacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Blackhat morphological operation to extract dark hair contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Intensify and create a binary mask of the hair lines
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint the original image using the hair mask boundaries
        clean_img = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
        return clean_img


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['image_id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        # Read image securely
        if not os.path.exists(img_path):
            logger.error(f"Image path missing: {img_path}")
            raise FileNotFoundError(f"Could not locate image: {img_path}")
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Clean image if flag is true
        if self.clean_hair:
            image = self._remove_hair(image)
            
        # Extract label index
        label_str = row['dx']
        label = self.label_map[label_str]
        
        # Apply Albumentations (Geometric/Photometric) Pipeline
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_transforms():
        """Returns standardized ImageNet transformations for Train and Validation splits."""
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
        return train_transform, val_transform