import pandas as pd
from tqdm import tqdm
import os
import json
import asyncio
import openai
import time
import re
from typing import List, Dict, Any, Tuple, Optional

# 导入提示词
from kg_extraction_prompts import ENTITY_EXTRACTION_PROMPT, RELATIONSHIP_EXTRACTION_PROMPT

from openai import OpenAI

class KnowledgeGraphExtractor:
    """知识图谱提取器"""
    
    def __init__(self, api_key=None, model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        """
        初始化知识图谱提取器
        
        Args:
            api_key: SiliconCloud API密钥
            model: 使用的模型名称
        """
        self.model = model
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(4)  # 限制并发请求数
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.siliconflow.cn/v1"  # SiliconCloud API地址
        )
        # 添加实体类型映射
        self.entity_types = {
            "人物": "Person",
            "组织": "Organization",
            "地点": "Location",
            "时间": "Time",
            "事件": "Event",
            "产品": "Product",
            "技术": "Technology",
            "船舶": "Ship",
            "项目": "Project"
        }

    async def call_openai_api(self, prompt: str) -> str:
        """调用SiliconCloud API"""
        async with self.semaphore:
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个专业的知识图谱提取助手，擅长从海事和船舶相关文本中提取实体和关系。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,  # 降低温度以获得更确定性的结果
                        max_tokens=2500
                    )
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"调用SiliconCloud API时出错: {e}")
                await asyncio.sleep(2)
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "你是一个专业的知识图谱提取助手，擅长从海事和船舶相关文本中提取实体和关系。"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=2500
                        )
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"重试调用SiliconCloud API时出错: {e}")
                    return ""
    
    def _extract_json_from_response(self, response: str) -> str:
        """从响应中提取JSON字符串"""
        try:
            # 尝试使用正则表达式提取JSON部分
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # 如果没有找到markdown格式的JSON，尝试直接解析整个响应
                json_str = response.strip()
                
            # 验证JSON是否有效
            json.loads(json_str)
            return json_str
        except Exception as e:
            print(f"JSON提取错误: {e}")
            # 尝试修复常见的JSON格式问题
            try:
                # 替换单引号为双引号
                fixed_str = response.replace("'", "\"")
                # 尝试找到可能的JSON部分
                start_idx = fixed_str.find("[")
                end_idx = fixed_str.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = fixed_str[start_idx:end_idx]
                    # 验证修复后的JSON
                    json.loads(json_str)
                    return json_str
            except:
                pass
            
            # 如果所有尝试都失败，返回空JSON数组
            return "[]"
    
    async def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表，每个实体包含名称和类型
        """
        # 对长文本进行分段处理
        max_chunk_length = 4000
        if len(text) > max_chunk_length:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            all_entities = []
            for chunk in chunks:
                chunk_entities = await self._extract_entities_from_chunk(chunk)
                all_entities.extend(chunk_entities)
            
            # 去重
            unique_entities = []
            seen = set()
            for entity in all_entities:
                entity_key = (entity['entity'], entity['type'])
                if entity_key not in seen:
                    seen.add(entity_key)
                    unique_entities.append(entity)
            
            return unique_entities
        else:
            return await self._extract_entities_from_chunk(text)
    
    async def _extract_entities_from_chunk(self, text: str) -> List[Dict[str, str]]:
        """从文本块中提取实体"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        try:
            response = await self.call_openai_api(prompt)
            # 提取JSON部分
            json_str = self._extract_json_from_response(response)
            entities = json.loads(json_str)
            
            # 规范化实体类型
            for entity in entities:
                if 'type' in entity:
                    entity_type = entity['type']
                    # 将中文类型映射为英文类型
                    for cn_type, en_type in self.entity_types.items():
                        if cn_type in entity_type:
                            entity['type'] = en_type
                            break
            
            return entities
        except Exception as e:
            print(f"提取实体时出错: {e}")
            return []
    
    async def extract_relations(self, text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系列表，每个关系包含头实体、关系类型和尾实体
        """
        # 对长文本进行分段处理
        max_chunk_length = 4000
        if len(text) > max_chunk_length:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            all_relations = []
            for chunk in chunks:
                chunk_relations = await self._extract_relations_from_chunk(chunk, entities)
                all_relations.extend(chunk_relations)
            
            # 去重
            unique_relations = []
            seen = set()
            for relation in all_relations:
                relation_key = (relation['head'], relation['relation'], relation['tail'])
                if relation_key not in seen:
                    seen.add(relation_key)
                    unique_relations.append(relation)
            
            return unique_relations
        else:
            return await self._extract_relations_from_chunk(text, entities)
    
    async def _extract_relations_from_chunk(self, text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """从文本块中提取关系"""
        entities_str = "\n".join([f"- {e['entity']} (类型: {e['type']})" for e in entities])
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(text=text, entities=entities_str)
        
        try:
            response = await self.call_openai_api(prompt)
            # 提取JSON部分
            json_str = self._extract_json_from_response(response)
            relations = json.loads(json_str)
            
            # 验证关系中的实体是否在实体列表中
            entity_names = {e['entity'] for e in entities}
            valid_relations = []
            for relation in relations:
                head = relation.get('head', '')
                tail = relation.get('tail', '')
                
                # 检查头尾实体是否在实体列表中
                if head in entity_names and tail in entity_names:
                    valid_relations.append(relation)
                else:
                    # 尝试查找最接近的实体名称
                    if head not in entity_names:
                        closest = self._find_closest_entity(head, entity_names)
                        if closest:
                            relation['head'] = closest
                    
                    if tail not in entity_names:
                        closest = self._find_closest_entity(tail, entity_names)
                        if closest:
                            relation['tail'] = closest
                    
                    # 如果现在头尾实体都有效，添加到有效关系中
                    if relation['head'] in entity_names and relation['tail'] in entity_names:
                        valid_relations.append(relation)
            
            return valid_relations
        except Exception as e:
            print(f"提取关系时出错: {e}")
            return []
    
    def _find_closest_entity(self, name: str, entity_names: set) -> str:
        """查找最接近的实体名称"""
        # 如果实体名是另一个实体的子字符串，使用该实体
        for entity in entity_names:
            if name in entity:
                return entity
            if entity in name:
                return entity
        return ""
    
    async def extract_knowledge_graph(self, text: str) -> List[List[str]]:
        """
        从文本中提取知识图谱
        
        Args:
            text: 输入文本
            
        Returns:
            知识图谱，格式为三元组列表
        """
        # 提取实体
        entities = await self.extract_entities(text)
        
        # 如果没有提取到实体，返回空图谱
        if not entities:
            return []
        
        # 提取关系
        relations = await self.extract_relations(text, entities)
        
        # 转换为指定格式的三元组
        triples = []
        for relation in relations:
            if 'head' in relation and 'relation' in relation and 'tail' in relation:
                triple = [relation["head"], relation["relation"], relation["tail"]]
                triples.append(triple)
        
        return triples
    
    def extract_knowledge_graph_sync(self, text: str) -> List[List[str]]:
        """同步版本的知识图谱提取"""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 创建新的事件循环
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self.extract_knowledge_graph(text))
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(self.extract_knowledge_graph(text))

async def process_news_data(news_file: str, output_dir: str):
    """处理新闻数据，提取知识图谱"""
    if not os.path.exists(news_file):
        raise FileNotFoundError(f"文件 {news_file} 不存在，请确保articles.xlsx文件在正确的位置")
    
    # 读取Excel文件
    df = pd.read_excel(news_file)
    
    # 检查必要的列是否存在
    if 'raw_content' not in df.columns:
        raise ValueError("Excel文件中缺少'raw_content'列")
    
    # 创建保存知识图谱的目录
    kg_dir = os.path.join(output_dir, "knowledge_graphs")
    os.makedirs(kg_dir, exist_ok=True)
    
    # 初始化知识图谱提取器 - 直接使用硬编码的API密钥
    extractor = KnowledgeGraphExtractor(api_key="sk-afawtrywwlidcukrnweqlfydnpfdjvpjdbtpccgdzkrsblhn")
    
    all_kg_triples = []
    
    # 处理前10条新闻
    for idx, row in tqdm(list(df.iterrows())[:10], desc="处理前10条新闻"):
        news_data = str(row['raw_content'])
        if pd.isna(news_data) or not news_data.strip():
            continue
        
        print(f"正在处理第{idx+1}条新闻...")
        
        # 提取知识图谱
        kg_triples = await extractor.extract_knowledge_graph(news_data)
        
        if kg_triples:
            # 保存单条新闻的知识图谱
            kg_file = os.path.join(kg_dir, f"news_kg_{idx}.json")
            with open(kg_file, 'w', encoding='utf-8') as f:
                json.dump(kg_triples, f, ensure_ascii=False, indent=2)
            
            print(f"已保存知识图谱到 {kg_file}")
            print(f"提取了 {len(kg_triples)} 个三元组")
            
            # 添加到总的知识图谱
            all_kg_triples.extend(kg_triples)
    
    # 保存总的知识图谱（增量更新）
    all_kg_file = os.path.join(output_dir, "all_news_kg.json")
    
    # 检查是否已存在知识图谱文件
    existing_triples = []
    if os.path.exists(all_kg_file):
        try:
            with open(all_kg_file, 'r', encoding='utf-8') as f:
                existing_triples = json.load(f)
            print(f"已加载现有知识图谱，包含 {len(existing_triples)} 个三元组")
        except Exception as e:
            print(f"读取现有知识图谱时出错: {e}")
            existing_triples = []
    
    # 创建一个集合来存储已有的三元组，用于快速查找
    existing_triples_set = {tuple(triple) for triple in existing_triples}
    
    # 计算新增的三元组数量
    new_triples_count = 0
    
    # 增量更新知识图谱
    for triple in all_kg_triples:
        if tuple(triple) not in existing_triples_set:
            existing_triples.append(triple)
            existing_triples_set.add(tuple(triple))
            new_triples_count += 1
    
    # 保存更新后的知识图谱
    with open(all_kg_file, 'w', encoding='utf-8') as f:
        json.dump(existing_triples, f, ensure_ascii=False, indent=2)
    
    print(f"知识图谱已增量更新，新增 {new_triples_count} 个三元组，总计 {len(existing_triples)} 个三元组")
    print(f"总共提取了 {len(all_kg_triples)} 个三元组")
    print(f"总的知识图谱已保存到 {all_kg_file}")
    
    return all_kg_triples

def main():
    # 使用当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 读取Excel文件中的新闻数据（与脚本在同一目录）
    news_file = os.path.join(current_dir, "articles.xlsx")
    
    # 保存数据集的路径（也在同一目录）
    output_dir = current_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理新闻数据
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    kg_triples = loop.run_until_complete(process_news_data(news_file, output_dir))
    
    print("知识图谱提取完成！")

if __name__ == "__main__":
    main()