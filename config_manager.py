# config_manager.py
# -*- coding: utf-8 -*-
import json
import os
import threading
from llm_adapters import create_llm_adapter
from embedding_adapters import create_embedding_adapter


DEFAULT_CONFIG = {
    "current_llm_interface": "OpenAI",
    "current_embedding_interface": "OpenAI",
    "llm_configs": {
        "DeepSeek": {
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat"
        },
        "OpenAI": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4"
        },
        "Azure OpenAI": {
            "api_key": "",
            "base_url": "https://[az].openai.azure.com/openai/deployments/[model]/chat/completions?api-version=2024-08-01-preview",
            "model_name": ""
        },
        "Ollama": {
            "api_key": "",
            "base_url": "http://localhost:11434/v1",
            "model_name": "llama2"
        },
        "ML Studio": {
            "api_key": "",
            "base_url": "http://localhost:1234/v1",
            "model_name": ""
        },
        "Gemini": {
            "api_key": "",
            "base_url": "",
            "model_name": "gemini-pro"
        }
    },
    "embedding_configs": {
        "DeepSeek": {
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "model_name": "deepseek-embedding"
        },
        "OpenAI": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model_name": "text-embedding-ada-002"
        },
        "Azure OpenAI": {
            "api_key": "",
            "base_url": "https://[az].openai.azure.com/openai/deployments/[model]/embeddings?api-version=2023-05-15",
            "model_name": ""
        },
        "Ollama": {
            "api_key": "",
            "base_url": "http://localhost:11434/api",
            "model_name": "nomic-embed-text"
        },
        "ML Studio": {
            "api_key": "",
            "base_url": "http://localhost:1234/v1",
            "model_name": ""
        },
        "Gemini": {
            "api_key": "",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
            "model_name": "text-embedding-004"
        }
    },
    "temperature": 0.7,
    "max_tokens": 8192,
    "timeout": 600,
    "embedding_retrieval_k": 4,
    "topic": "",
    "genre": "玄幻",
    "num_chapters": 10,
    "word_number": 3000,
    "filepath": "",
    "chapter_num": "1",
    "user_guidance": "",
    "characters_involved": "",
    "key_items": "",
    "scene_location": "",
    "time_constraint": ""
}


def load_config(config_file: str) -> dict:
    """从指定的 config_file 加载配置，若不存在则创建默认配置文件。"""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取配置文件出错: {str(e)}，将使用默认配置")
            return create_default_config(config_file)
    else:
        return create_default_config(config_file)


def create_default_config(config_file: str) -> dict:
    """创建默认配置文件"""
    try:
        # 如果config_file包含路径，则创建目录
        dirname = os.path.dirname(config_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # 写入默认配置
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
        print(f"已创建默认配置文件: {config_file}")
        return DEFAULT_CONFIG
    except Exception as e:
        print(f"创建默认配置文件失败: {str(e)}")
        return DEFAULT_CONFIG


def save_config(config_data: dict, config_file: str) -> None:
    """保存配置到文件"""
    try:
        # 如果config_file包含路径，则创建目录
        dirname = os.path.dirname(config_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存配置文件失败: {str(e)}")


def test_llm_config(interface_format, api_key, base_url, model_name, temperature, max_tokens, timeout, log_func, handle_exception_func):
    """测试当前的LLM配置是否可用"""
    def task():
        try:
            log_func("开始测试LLM配置...")
            llm_adapter = create_llm_adapter(
                interface_format=interface_format,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

            test_prompt = "Please reply 'OK'"
            response = llm_adapter.invoke(test_prompt)
            if response:
                log_func("✅ LLM配置测试成功！")
                log_func(f"测试回复: {response}")
            else:
                log_func("❌ LLM配置测试失败：未获取到响应")
        except Exception as e:
            log_func(f"❌ LLM配置测试出错: {str(e)}")
            handle_exception_func("测试LLM配置时出错")

    threading.Thread(target=task, daemon=True).start()

def test_embedding_config(api_key, base_url, interface_format, model_name, log_func, handle_exception_func):
    """测试当前的Embedding配置是否可用"""
    def task():
        try:
            log_func("开始测试Embedding配置...")
            embedding_adapter = create_embedding_adapter(
                interface_format=interface_format,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )

            test_text = "测试文本"
            embeddings = embedding_adapter.embed_query(test_text)
            if embeddings and len(embeddings) > 0:
                log_func("✅ Embedding配置测试成功！")
                log_func(f"生成的向量维度: {len(embeddings)}")
            else:
                log_func("❌ Embedding配置测试失败：未获取到向量")
        except Exception as e:
            log_func(f"❌ Embedding配置测试出错: {str(e)}")
            handle_exception_func("测试Embedding配置时出错")

    threading.Thread(target=task, daemon=True).start()