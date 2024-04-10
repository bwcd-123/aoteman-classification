from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images_path:list, labels:list, transform=None):
        super(ImageDataset, self).__init__()
        self.image_path = images_path
        self.image_labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.image_path)
    

    def __getitem__(self, index):
        img = Image.open(self.image_path[index])
        # 判断图片是否为彩色格式
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if img.mode != "RGB":
            raise ValueError("image: {} isn't RGB mode.".format(self.image_path[index]))
        
        label = self.image_labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label
    

    @staticmethod
    def collate_fn(patch):
        images, labels = tuple(zip(*patch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
        