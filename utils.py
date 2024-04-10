import os
import shutil
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from thop import profile, clever_format
import tqdm

from mydataset import ImageDataset

dict = {
    'dijia':0,
    'jieke':1,
    'saiwen':2,
    'tailuo':3
}


def remove_oldest_item(path:str, number:int=10):  
    """当文件数量达到numbers时，删除修改时间最早的文件或文件夹"""
    # 获取目录下所有文件和文件夹的路径
    items = [os.path.join(path, item) for item in os.listdir(path)]
    # 过滤出文件和文件夹，并分别按照修改时间排序
    files = sorted([(f, os.path.getmtime(f)) for f in items if os.path.isfile(f)], key=lambda x: x[1])
    folders = sorted([(d, os.path.getmtime(d)) for d in items if os.path.isdir(d)], key=lambda x: x[1])
    
    # 合并文件和文件夹列表，并按修改时间排序
    items_with_mtime = files + folders
    # 如果文件数量未达到number，则无需删除
    if len(items_with_mtime) < number:
        return
    if items_with_mtime:
        # 删除修改时间最早的文件或文件夹
        oldest_item_path, mtime = items_with_mtime[0]
        try:
            if os.path.isfile(oldest_item_path):
                os.remove(oldest_item_path)
                print(f"delete a file: {oldest_item_path}")
            elif os.path.isdir(oldest_item_path):
                shutil.rmtree(oldest_item_path)
                print(f"delete a directory: {oldest_item_path}")
        except Exception as e:
            print(f'fail to delete "{oldest_item_path}": {e}')


def read_aoteman_data(root:str, train:bool=True, transform=None, **kwargs):
    """
    Read aoteman dataset. If train, return train dataset. Else, return eval dataset.
    """
    test_root = os.path.join(root, "predict_demo.jpg")
    if not os.path.exists(test_root):
        raise Exception(f"error root: '{root}'")
    
    if train:
        root = os.path.join(root, "train")
    else:
        root = os.path.join(root, "eval")
    
    categories = os.listdir(root)
    images = []
    labels = []

    for category in categories:
        branch = os.path.join(root, category)
        label = dict[category]
        for image_name in os.listdir(branch):
            image_path = os.path.join(branch, image_name)
            images.append(image_path)
            labels.append(label)
        
    return images, labels
        

def train_one_epoch(model, optimizer, data_loader, loss_fn, device):
    """
    Train the model. Return the average loss and accuracy.

    Parameters:
        model(torch.nn.Module): model to train.
        data_loader(torch.utils.data.Dataloader): data loader.
        loss_fn: loss function.
        device: device to train.
    """
    model.train()
    optimizer.zero_grad()

    batch_len = len(data_loader)
    data_len = 0.
    total_loss = 0.
    rights = 0.
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        data_len += len(labels)

        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach().item()
        rights += (predictions.argmax(dim=1) == labels).sum().item()
    
    return total_loss / batch_len, rights / data_len


def eval(model, data_loader, loss_fn, device):
    """
    Eval the model. Return the average loss and accuracy.

    Parameters:
        model(torch.nn.Module): model to eval.
        data_loader(torch.utils.data.Dataloader): data loader.
        loss_fn: loss function.
        device: device to eval.
    """
    model.eval()
    batch_len = len(data_loader)
    total_loss = 0.
    rights = 0.
    data_len = 0.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            data_len += len(labels)
            
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            
            total_loss += loss.detach().item()
            rights += (predictions.argmax(dim=1) == labels).sum().item()
            
    return total_loss / batch_len, rights / data_len


def show_L1Norm_vgg(model, save_path=None):
    """
    show L1_Norm in vgg11 model with graph.
    """
    index = 1
    plt.figure(figsize=(14, 6))
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # 计算当前卷积层每一个卷积核的L1范数
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = weight_copy.sum(axis=1).sum(axis=1).sum(axis=1)
            sort = L1_norm.argsort()
            L1_norm = L1_norm[sort]
            plt.subplot(2, 4, index)
            index += 1
            min_10 = int(len(L1_norm) / 10)
            min_20 = int(len(L1_norm) / 5)
            min_10 = torch.ones_like(L1_norm) * L1_norm[int(min_10)]
            min_20 = torch.ones_like(L1_norm) * L1_norm[int(min_20)]
            plt.plot(L1_norm, label="L1_norm")
            plt.plot(min_10, label="min 10%")
            plt.plot(min_20, label="min 20%")
            plt.xlabel("kernel index")
            plt.ylabel("L1_norm")
            plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def get_params_and_FLOPs(model, input_size):
    """
    get params and FLOPs of model.
    """
    input_tensor = torch.rand(input_size)
    FLOPs, params = profile(model, inputs=(input_tensor,))
    FLOPs, params = clever_format([FLOPs, params], "%.3f")
    return params, FLOPs


def get_mean_and_std(dataloader):
    """
    get mean and std of dataloader.
    """
    data, _ = next(iter(dataloader))

    # get the everage of the dataloader
    data_mean = torch.mean(data, dim=[0, 2, 3])

    means = torch.ones_like(data, dtype=torch.float32)
    means[:, 0, :, :] = data_mean[0]
    means[:, 1, :, :] = data_mean[1]
    means[:, 2, :, :] = data_mean[2]

    # get the std of the dataset
    data_std = torch.std(data, dim=[0, 2, 3])

    return data_mean, data_std


if __name__ == "__main__":
    train_list, train_labels = read_aoteman_data(root='../datasets/aoteman', train=True)
    test_list, test_labels = read_aoteman_data(root='../datasets/aoteman', train=False)
    datas = train_list + test_list
    labels = train_labels + test_labels
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    all_dataset = ImageDataset(datas, labels, transform=transform)
    dataloader = DataLoader(all_dataset, batch_size=1000)

    mean_data, std_data = get_mean_and_std(dataloader)
    print(mean_data, std_data)