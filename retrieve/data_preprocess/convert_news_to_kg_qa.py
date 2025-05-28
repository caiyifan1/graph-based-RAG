import spacy
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import os

# 如果没有安装spacy中文模型，请先运行:
# python -m spacy download zh_core_web_sm

# 加载中文NLP模型
try:
    nlp = spacy.load("zh_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "zh_core_web_sm"])
    nlp = spacy.load("zh_core_web_sm")

def extract_entities(text):
    """从文本中抽取实体"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "type": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities

def extract_relations(text, entities):
    """简单的关系抽取方法"""
    relations = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:
                # 简单的共现关系
                if abs(entity1["start"] - entity2["start"]) < 100:
                    relation_type = "相关"
                    relations.append({
                        "head": entity1["text"],
                        "relation": relation_type,
                        "tail": entity2["text"]
                    })
    return relations

def generate_qa_pairs(entities, relations):
    """根据实体和关系生成问答对"""
    qa_pairs = []
    
    # 为每个实体生成问题
    for entity in entities:
        related_relations = [r for r in relations if r["head"] == entity["text"] or r["tail"] == entity["text"]]
        if related_relations:
            if entity["type"] == "PERSON":
                question = f"谁是{entity['text']}?"
            elif entity["type"] == "ORG":
                question = f"{entity['text']}是什么组织?"
            elif entity["type"] == "GPE":
                question = f"{entity['text']}在哪里?"
            else:
                question = f"什么是{entity['text']}?"
                
            answer = entity["text"]
            q_entity = [entity["text"]]
            a_entity = [entity["text"]]
            graph = [(r["head"], r["relation"], r["tail"]) for r in related_relations]
            
            qa_pairs.append({
                "question": question,
                "answer": [answer],
                "q_entity": q_entity,
                "a_entity": a_entity,
                "graph": graph
            })
    
    return qa_pairs

def main():
    # 读取Excel文件中的新闻数据
    news_file = "d:\\pythonCode\\SubgraphRAG\\articles.xlsx"
    
    if not os.path.exists(news_file):
        raise FileNotFoundError(f"文件 {news_file} 不存在，请确保articles.xlsx文件在正确的位置")
    
    # 读取Excel文件
    df = pd.read_excel(news_file)
    
    # 检查必要的列是否存在
    if 'raw_content' not in df.columns:
        raise ValueError("Excel文件中缺少'raw_content'列")
    
    all_qa_pairs = []
    print("正在处理新闻数据...")
    
    # 使用tqdm显示处理进度
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理新闻"):
        news_data = str(row['raw_content'])
        if pd.isna(news_data) or not news_data.strip():
            continue
            
        # 抽取实体
        entities = extract_entities(news_data)
        if not entities:
            continue
            
        # 抽取关系
        relations = extract_relations(news_data, entities)
        if not relations:
            continue
            
        # 生成问答对
        qa_pairs = generate_qa_pairs(entities, relations)
        if qa_pairs:
            # 为每个问答对添加唯一ID
            for i, qa in enumerate(qa_pairs):
                qa["id"] = f"news_qa_{idx}_{i}"
            all_qa_pairs.extend(qa_pairs)
    
    print(f"总共生成了 {len(all_qa_pairs)} 个问答对")
    
    if not all_qa_pairs:
        raise ValueError("没有生成任何有效的问答对，请检查数据质量")
    
    # 创建Hugging Face数据集
    dataset = Dataset.from_list(all_qa_pairs)
    
    # 分割数据集
    train_test_split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_val_split = train_test_split["train"].train_test_split(test_size=0.25, shuffle=True, seed=42)
    
    # 最终数据集
    final_dataset = {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"]
    }
    
    # 保存数据集
    dataset_path = "d:\\pythonCode\\SubgraphRAG\\data_files\\news_kg"
    os.makedirs(dataset_path, exist_ok=True)
    
    for split, ds in final_dataset.items():
        ds.save_to_disk(os.path.join(dataset_path, split))
        print(f"{split} 集合大小: {len(ds)} 个问答对")
    
    print(f"\n数据集已保存到 {dataset_path}")
    print("现在您可以修改emb.py文件，添加对新数据集的支持")

if __name__ == "__main__":
    main()