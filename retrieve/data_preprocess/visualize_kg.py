import json
import os
import sys
import argparse
from pyvis.network import Network
import pandas as pd
import random

def load_knowledge_graph(file_path):
    """加载知识图谱数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        kg_triples = json.load(f)
    return kg_triples

def create_color_map(triples):
    """为不同类型的实体创建颜色映射"""
    # 提取所有实体
    entities = set()
    for triple in triples:
        entities.add(triple[0])
        entities.add(triple[2])
    
    # 为实体分配随机颜色
    colors = {}
    for entity in entities:
        # 生成柔和的颜色
        r = random.randint(100, 200)
        g = random.randint(100, 200)
        b = random.randint(100, 200)
        colors[entity] = f'#{r:02x}{g:02x}{b:02x}'
    
    return colors

def visualize_knowledge_graph(kg_triples, output_file, title="知识图谱可视化"):
    """可视化知识图谱"""
    try:
        # 创建网络图
        net = Network(height="750px", width="100%", bgcolor="#ffffff", 
                     font_color="black", notebook=False, directed=True)
        
        # 设置物理布局
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.001,
            damping=0.09,
            overlap=0.1
        )
        
        # 创建颜色映射
        color_map = create_color_map(kg_triples)
        added_nodes = set()
        
        # 添加节点和边
        for head, relation, tail in kg_triples:
            if head not in added_nodes:
                net.add_node(head, label=head, title=head, color=color_map.get(head, "#97c2fc"))
                added_nodes.add(head)
            
            if tail not in added_nodes:
                net.add_node(tail, label=tail, title=tail, color=color_map.get(tail, "#97c2fc"))
                added_nodes.add(tail)
            
            net.add_edge(head, tail, label=relation, title=relation)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存为HTML文件
        net.save_graph(output_file)
        print(f"知识图谱已可视化并保存到 {output_file}")
        return True
        
    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")
        return False

def visualize_single_file(input_file, output_file=None):
    """可视化单个知识图谱文件"""
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    # 如果未指定输出文件，则使用输入文件名生成输出文件名
    if output_file is None:
        output_dir = os.path.dirname(input_file)
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_visualization.html")
    
    print(f"正在可视化 {input_file}...")
    kg_triples = load_knowledge_graph(input_file)
    
    if kg_triples:
        visualize_knowledge_graph(kg_triples, output_file, title=f"知识图谱 - {os.path.basename(input_file)}")
        print(f"成功可视化 {input_file}，包含 {len(kg_triples)} 个三元组")
        return True
    else:
        print(f"警告: {input_file} 不包含任何三元组")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='知识图谱可视化工具')
    parser.add_argument('--input', '-i', type=str, help='输入的JSON知识图谱文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出的HTML可视化文件路径（可选）')
    parser.add_argument('--dir', '-d', type=str, help='包含多个JSON知识图谱文件的目录路径')
    
    args = parser.parse_args()
    
    # 如果指定了单个文件
    if args.input:
        visualize_single_file(args.input, args.output)
        return
    
    # 如果指定了目录
    if args.dir:
        kg_dir = args.dir
    else:
        # 默认目录
        kg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_files", "knowledge_graphs")
    
    if not os.path.exists(kg_dir):
        print(f"知识图谱目录 {kg_dir} 不存在")
        return
    
    # 创建可视化输出目录
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_files", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 处理每个单独的知识图谱文件
    success_count = 0
    for file in os.listdir(kg_dir):
        if file.endswith(".json"):
            kg_file = os.path.join(kg_dir, file)
            output_file = os.path.join(vis_dir, f"{file.replace('.json', '.html')}")
            
            if visualize_single_file(kg_file, output_file):
                success_count += 1
    
    print(f"总共成功可视化了 {success_count} 个知识图谱文件")

if __name__ == "__main__":
    main()