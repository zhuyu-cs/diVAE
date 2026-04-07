"""
batch_train.py
批量训练生成的SCN数据

目录结构：
results/
├── repeat1/
│   ├── all_neuron_20210916/
│   ├── all_neuron_20210918/
│   ├── all_neuron_20210922/
│   ├── all_neuron_20220726/
│   ├── all_neuron_20220728/
│   └── all_neuron_20220730/
├── repeat2/
│   └── ...
...
"""

import numpy as np
import argparse
import os
import time
import datetime
from TraceContrast_model import TraceContrast
import datautils
from utils import init_dl_program
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy.io as scio


# 6只SCN的标识符
MICE_LIST = [
    'all_neuron_20210916',
    'all_neuron_20210918',
    'all_neuron_20210922',
    'all_neuron_20220726',
    'all_neuron_20220728',
    'all_neuron_20220730'
]


def export_ply_with_label(out, points, colors):
    """导出带标签的点云文件"""
    with open(out, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex ' + str(points.shape[0]) + '\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            cur_color = colors[i, :]
            f.write('%f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2],
                int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)
            ))


def train_single_scn(train_data, poi, output_dir, config, device):
    """
    训练单个SCN并保存结果
    
    Args:
        train_data: 训练数据 (N, num_trials, trial_length)
        poi: 位置信息 (N, 3)
        output_dir: 输出目录
        config: 训练配置
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    t = time.time()
    
    # 创建模型
    model = TraceContrast(
        input_dims=train_data.shape[-1],
        device=device,
        batch_size=config['batch_size'],
        lr=config['lr'],
        output_dims=config['repr_dims'],
        max_train_length=config['max_train_length']
    )
    
    # 训练
    loss_log = model.fit(
        train_data,
        n_epochs=config['epochs'],
        n_iters=config['iters'],
        verbose=True
    )
    
    # 保存模型
    model.save(f'{output_dir}/model.pkl')
    
    t = time.time() - t
    print(f"  Training time: {datetime.timedelta(seconds=t)}")
    
    # 保存loss曲线
    np.save(f'{output_dir}/loss_log.npy', np.array(loss_log))
    
    # 获取embedding
    train_repr = model.encode(train_data)
    
    from sklearn.preprocessing import normalize
    embeddings = np.reshape(train_repr, (train_repr.shape[0], train_repr.shape[1] * train_repr.shape[2]))
    embeddings = normalize(embeddings)
    
    # 保存embeddings
    scio.savemat(f'{output_dir}/embedding.mat', {'emb': embeddings})
    
    # t-SNE降维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(embeddings)
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)
    
    scio.savemat(f'{output_dir}/tsne.mat', {'tsne': x_norm})
    
    # 无监督聚类 (1-5类)
    for num_classes in range(1, 6):
        if num_classes == 1:
            y = np.zeros((embeddings.shape[0],), dtype=int)
        else:
            kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings)
            y = kmeans.labels_
        
        scio.savemat(f'{output_dir}/class_order_{num_classes}.mat', {'order': y})
        
        # 计算类别均值用于排序
        class_mean = np.zeros((num_classes, 1))
        for k in range(num_classes):
            class_order = np.where(y == k)
            tmp_pos = x_norm[class_order, 0]
            class_mean[k] = np.mean(tmp_pos)
        new_mean = np.argsort(class_mean, axis=0)
        
        # 绘制t-SNE图
        plt.figure(figsize=(6, 6))
        for i in range(x_norm.shape[0]):
            for j in range(num_classes):
                if new_mean[j] == y[i]:
                    color_order = j
            plt.scatter(
                x_norm[i, 0], x_norm[i, 1], marker='.', color=plt.cm.Set1(color_order)
            )
        if num_classes == 1:
            plt.title(f'{num_classes} Cluster', size=20, fontweight='bold')
        else:
            plt.title(f'{num_classes} Clusters', size=20, fontweight='bold')
        plt.yticks(fontproperties='Arial', size=20, fontweight='bold')
        plt.xticks(fontproperties='Arial', size=20, fontweight='bold')
        plt.savefig(f'{output_dir}/tsne_{num_classes}.eps', dpi=400)
        plt.savefig(f'{output_dir}/tsne_{num_classes}.png', dpi=200)  # 额外保存png便于查看
        plt.close()
        
        # 生成点云数据
        points = poi.cpu().numpy() if hasattr(poi, 'cpu') else poi
        colors = np.zeros_like(points)
        for i in range(colors.shape[0]):
            if y[i] < 9:
                colors[i] = np.array(plt.cm.Set1(y[i])[:3])
            elif y[i] >= 9 and y[i] < 17:
                colors[i] = np.array(plt.cm.Set2(y[i] - 9)[:3])
            else:
                colors[i] = np.array(plt.cm.Set3(y[i] - 17)[:3])
        
        export_ply_with_label(f'{output_dir}/{num_classes}_clusters.ply', points, colors)
    
    return loss_log


def main():
    parser = argparse.ArgumentParser(description='Batch training for generated SCN data')
    parser.add_argument('--input_dir', type=str, default='./generated/',
                        help='Directory containing generated pkl files')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='Output directory for results')
    parser.add_argument('--task', type=str, default='standard',
                        help='Task type (standard, time-sample, etc.)')
    parser.add_argument('--num_repeats', type=int, default=5,
                        help='Number of repeats to process')
    
    # 训练参数
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--repr_dims', type=int, default=16, help='Representation dimension')
    parser.add_argument('--max_train_length', type=int, default=10000, help='Max train length')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--iters', type=int, default=None, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # 初始化设备
    device = init_dl_program(args.gpu, seed=args.seed)
    
    # 训练配置
    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'repr_dims': args.repr_dims,
        'max_train_length': args.max_train_length,
        'epochs': args.epochs,
        'iters': args.iters
    }
    
    print("=" * 70)
    print("Batch Training for Generated SCN Data")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Task: {args.task}")
    print(f"Number of repeats: {args.num_repeats}")
    print(f"Mice list: {MICE_LIST}")
    print(f"Config: {config}")
    print("=" * 70)
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录总体进度
    total_tasks = args.num_repeats * len(MICE_LIST)
    current_task = 0
    total_start_time = time.time()
    
    # 遍历每个repeat
    for repeat_idx in range(1, args.num_repeats + 1):
        print(f"\n{'#' * 70}")
        print(f"# Processing Repeat {repeat_idx}/{args.num_repeats}")
        print(f"{'#' * 70}")
        
        # pkl文件路径
        pkl_path = os.path.join(args.input_dir, f'generated_activity_repeat{repeat_idx}.pkl')
        
        if not os.path.exists(pkl_path):
            print(f"  WARNING: {pkl_path} not found, skipping...")
            continue
        
        # 创建repeat目录
        repeat_dir = os.path.join(args.output_dir, f'repeat{repeat_idx}')
        os.makedirs(repeat_dir, exist_ok=True)
        
        # 遍历每只SCN
        for mouse_key in MICE_LIST:
            current_task += 1
            print(f"\n  [{current_task}/{total_tasks}] Processing: {mouse_key}")
            print(f"  " + "-" * 50)
            
            # 加载数据
            try:
                train_data, poi = datautils.load_generated_SCN(pkl_path, mouse_key, args.task)
                print(f"    Data shape: {train_data.shape}")
                print(f"    Position shape: {poi.shape}")
            except Exception as e:
                print(f"    ERROR loading data: {e}")
                continue
            
            # 输出目录
            mouse_output_dir = os.path.join(repeat_dir, mouse_key)
            
            # 训练
            try:
                train_single_scn(train_data, poi, mouse_output_dir, config, device)
                print(f"    Results saved to: {mouse_output_dir}")
            except Exception as e:
                print(f"    ERROR during training: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 70}")
    print(f"All tasks completed!")
    print(f"Total time: {datetime.timedelta(seconds=total_time)}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()