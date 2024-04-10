import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
import swanlab
import time
from datetime import datetime

from models.process import AddNoise
from utils import get_model, read_aoteman_data, train_one_epoch_single_cuda, eval_single_cuda, formate_abs_path
from mydataset import ImageDataset


# 奥特曼数据集的方差[0.4654, 0.4327, 0.4739]，标准差[0.2775, 0.2903, 0.3157]
# the mean of dataset of aoteman is [0.4654, 0.4327, 0.4739], and the std is [0.2775, 0.2903, 0.3157]
train_trainsform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4654, 0.4327, 0.4739], [0.2775, 0.2903, 0.3157]),
    #  AddNoise(),
])
eval_trainsform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4654, 0.4327, 0.4739], [0.2775, 0.2903, 0.3157]),
])

def main(opt):
    logger = swanlab.init(
        experiment_name=opt.model,
        description="train pvt model",
        config=opt,
        logdir="./logs",
    )
    log_path = formate_abs_path(logger.settings.run_dir).replace('\\','/')
    # 设置全局随机种子
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)  # 可选：为CUDA设置一个特定的种子
        torch.cuda.manual_seed_all(opt.seed)  # 为所有CUDA设备设置相同的种子
        
    model = get_model(opt.model)
    model = model.to(opt.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=-1e-5)

    train_paths, train_labels = read_aoteman_data(opt.data_path, True)
    eval_paths, eval_labels = read_aoteman_data(opt.data_path, False)
    train_dataset = ImageDataset(train_paths, train_labels, train_trainsform)
    eval_dataset = ImageDataset(eval_paths, eval_labels, eval_trainsform)
    pin_memory = True if opt.device == "cuda" else False
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, collate_fn=ImageDataset.collate_fn, pin_memory=pin_memory)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False,
                                 num_workers=opt.num_workers, collate_fn=ImageDataset.collate_fn, pin_memory=pin_memory)
    
    for epoch in range(opt.epochs):
        start_time = time.perf_counter()
        train_loss, train_acc = train_one_epoch_single_cuda(model, optimizer, train_dataloader, loss_fn, opt.device)
        logger.log({"train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}.")
        scheduler.step()
        eval_loss, eval_acc = eval_single_cuda(model, eval_dataloader, loss_fn, opt.device)
        logger.log({"eval_loss": eval_loss, "eval_acc": eval_acc})
        print(f"Epoch {epoch}, eval_loss: {eval_loss:.4f}, eval_acc: {eval_acc:.3f}.")
        end_time = time.perf_counter()
        print(f"Epoch {epoch} takes {end_time - start_time:.2f}s.\n")

    # 获取当前时间
    current_time = datetime.now()

    # 将当前时间格式化为字符串，例如：'04101500'
    time_str = current_time.strftime("%m%d%H%M")   
    try:
        torch.save(model.state_dict(), f"weights/PVT/{opt.model}_{time_str}.pth")
    except:
        print("save model failed!")
    else:
        print(f"save model 'weights/PVT/{opt.model}_{time_str}.pth' successfully!")

# python train_single_cuda.py --data_path ../datasets/aoteman  --epochs 1
if __name__ == '__main__':
    arg = argparse.ArgumentParser(description="train pvt model")
    arg.add_argument("--model", type=str, default="pvt_v2_b0",
                     choices=['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li'],
                     help="function name")
    arg.add_argument("--data_path", type=str, help="path of dataset")
    arg.add_argument("--batch_size", type=int, default=10, help="batch size")
    arg.add_argument("--epochs", type=int, default=100, help="epochs")
    arg.add_argument("--lr", type=float, default=0.001, help="learning rate")
    arg.add_argument("--seed", type=int, default=42, help="random seed")
    arg.add_argument("--device", type=str, default="cuda", help="device")
    arg.add_argument("--num_workers", type=int, default=0, help="num of workers")
    opt = arg.parse_args()
    main(opt)
