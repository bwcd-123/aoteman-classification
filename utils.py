import os
import shutil


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


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, optimizer, data_loader, loss_fn, device):
    pass