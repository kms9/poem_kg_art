import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

class PoemsSearcher:
    def __init__(self):
        self.encoder = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.poems: Optional[pd.DataFrame] = None
        self.index = None
        
    def load_poems(self, file_pattern: Union[str, Path]):
        """加载多个JSON格式的诗词文件
        
        Args:
            file_pattern: 文件匹配模式，如 'poem/poet.*' 或 'poem/*.json'
        """
        all_poems = []
        files = list(glob.glob(str(file_pattern)))
        
        for file_path in tqdm(files, desc="加载诗词文件"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    poems_data = json.load(f)
                    all_poems.extend(poems_data)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
                continue
        
        # 预处理数据
        processed_poems = []
        for poem in all_poems:
            processed_poems.append({
                'title': poem.get('rhythmic', poem.get('title', '')),
                'author': poem.get('author', '无名氏'),
                'content': ' '.join(poem['paragraphs']),
            })
            
        self.poems = pd.DataFrame(processed_poems)
        print(f"总共加载了 {len(self.poems)} 首诗词")
        
    def search(self, query, top_k=20, threshold=0.1):
        """搜索诗词并返回结果
        
        Returns:
            List[Dict]: 每个字典包含完整的诗词信息，包括分段的paragraphs
        """
        # 获取查询向量
        query_vector = self.encoder.encode([query])
        
        # 搜索结果
        distances, indices = self.index.search(query_vector, k=top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # 直接使用距离作为相似度（距离越小越相似）
            similarity = float(distances[0][i])
            
            poem = self.poems.iloc[idx]
            results.append({
                'title': poem.get('rhythmic', poem.get('title', '')),
                'author': poem['author'],
                'content': poem['content'],
                'similarity': similarity,  # 直接使用原始距离
                # 'debug_info': {  # 添加调试信息
                #     'index': idx,
                #     'raw_distance': float(distances[0][i])
                # }
            })
        
        # 按距离升序排序（距离越小越相似）
        results.sort(key=lambda x: x['similarity'])
        
        return results
        
    def save_index(self, directory: Union[str, Path]):
        """保存索引和数据到指定目录
        
        Args:
            directory: 保存目录路径
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        index_path = directory / 'poems.index'
        faiss.write_index(self.index, str(index_path))
        
        # 保存DataFrame和其他必要数据
        data = {
            'poems': self.poems,
            'index_type': type(self.index).__name__
        }
        with open(directory / 'poems.pkl', 'wb') as f:
            pickle.dump(data, f)
            
        print(f"索引和数据已保存到: {directory}")
        
    def load_index(self, directory: Union[str, Path]):
        """从指定目录加载索引和数据
        
        Args:
            directory: 索引所在目录路径
        """
        directory = Path(directory)
        
        # 加载索引
        index_path = directory / 'poems.index'
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
            
        self.index = faiss.read_index(str(index_path))
        
        # 加载DataFrame和其他数据
        with open(directory / 'poems.pkl', 'rb') as f:
            data = pickle.load(f)
            self.poems = data['poems']
            
        print(f"已加载索引，包含 {self.index.ntotal} 个向量")
        
    def build_index(self, save_dir: Optional[Union[str, Path]] = None):
        """构建向量索引，可选择直接保存
        
        Args:
            save_dir: 如果提供，将在构建后保存索引到该目录
        """
        if self.poems is None:
            raise ValueError("请先加载诗词数据")
            
        print("开始构建向量索引...")
        poems_text = self.poems['content'].tolist()
        
        # 批量处理向量生成
        batch_size = 1000
        all_embeddings = []
        
        for i in tqdm(range(0, len(poems_text), batch_size), desc="生成向量"):
            batch = poems_text[i:i + batch_size]
            embeddings = self.encoder.encode(batch)
            all_embeddings.append(embeddings)
            
        embeddings = np.vstack(all_embeddings)
        dimension = embeddings.shape[1]
        
        # 根据数据量选择索引类型
        if len(self.poems) > 10000:
            nlist = min(4096, int(np.sqrt(len(self.poems))))
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            print("训练IVF索引...")
            self.index.train(embeddings.astype('float32'))
        else:
            self.index = faiss.IndexFlatL2(dimension)
            
        self.index.add(embeddings.astype('float32'))
        print(f"索引构建完成，包含 {self.index.ntotal} 个向量")
        
        # 如果提供了保存目录，直接保存
        if save_dir:
            self.save_index(save_dir)

    @classmethod
    def initialize(cls, poems_pattern: Union[str, List[str]], index_dir: str, force_rebuild: bool = False):
        """便捷初始化方法
        
        Args:
            poems_pattern: 诗词文件匹配模式,可以是单个字符串或字符串列表
            index_dir: 索引目录
            force_rebuild: 是否强制重建索引
        
        Returns:
            PoemRetriever: 初始化好的检索器实例
        """
        retriever = cls()
        index_dir = Path(index_dir)
        
        if not force_rebuild and index_dir.exists():
            try:
                print("尝试加载现有索引...")
                retriever.load_index(index_dir)
                return retriever
            except Exception as e:
                print(f"加载现有索引失败: {e}")
                print("将重新构建索引...")
        
        # 构建新索引
        if isinstance(poems_pattern, str):
            poems_pattern = [poems_pattern]
            
        for pattern in poems_pattern:
            retriever.load_poems(pattern)
            
        retriever.build_index(save_dir=index_dir)
        return retriever