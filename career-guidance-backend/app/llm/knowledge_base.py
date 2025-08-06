from typing import Dict, List, Any
import os
import json


class KnowledgeBaseManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def load_knowledge_base(self) -> Dict[str, List[str]]:
        knowledge_data = {}
        for category in os.listdir(self.base_dir):
            category_path = os.path.join(self.base_dir, category)
            if os.path.isdir(category_path):
                knowledge_data[category] = self._load_category_files(
                    category_path)
        return knowledge_data

    def _load_category_files(self, category_path: str) -> List[str]:
        texts = []
        for filename in os.listdir(category_path):
            if filename.endswith('.txt'):
                with open(os.path.join(category_path, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
        return texts

    def save_knowledge_base(self, data: Dict[str, List[str]]):
        for category, texts in data.items():
            category_path = os.path.join(self.base_dir, category)
            os.makedirs(category_path, exist_ok=True)
            for i, text in enumerate(texts):
                with open(os.path.join(category_path, f'doc_{i}.txt'), 'w', encoding='utf-8') as file:
                    file.write(text)

    def get_categories(self) -> List[str]:
        return os.listdir(self.base_dir) if os.path.exists(self.base_dir) else []

    def clear_knowledge_base(self):
        for category in self.get_categories():
            category_path = os.path.join(self.base_dir, category)
            for filename in os.listdir(category_path):
                os.remove(os.path.join(category_path, filename))
            os.rmdir(category_path)
