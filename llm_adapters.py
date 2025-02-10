# llm_adapters.py
# -*- coding: utf-8 -*-
import logging
import requests
from typing import Optional
from langchain_openai import ChatOpenAI

def ensure_openai_base_url_has_v1(url: str) -> str:
    import re
    url = url.strip()
    if not url:
        return url
    if not re.search(r'/v\d+$', url):
        if '/v1' not in url:
            url = url.rstrip('/') + '/v1'
    return url

class BaseLLMAdapter:
    """
    统一的 LLM 接口基类，为不同后端（OpenAI、Ollama、ML Studio 等）提供一致的方法签名。
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement .invoke(prompt) method.")

class DeepSeekAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = ensure_openai_base_url_has_v1(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from DeepSeekAdapter.")
            return ""
        return response.content

class OpenAIAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = ensure_openai_base_url_has_v1(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OpenAIAdapter.")
            return ""
        return response.content

class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama 同样有一个 OpenAI-like /v1/chat 接口，可直接使用 ChatOpenAI。
    但是通常 Ollama 默认本地服务在 http://localhost:11434，如果符合OpenAI风格即可直接传参。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = ensure_openai_base_url_has_v1(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OllamaAdapter.")
            return ""
        return response.content

class MLStudioAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = ensure_openai_base_url_has_v1(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from MLStudioAdapter.")
            return ""
        return response.content

class RAGFlowAdapter(BaseLLMAdapter):
    """
    适配RAGFlow API接口
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 初始化会话相关属性
        self.chat_id = None
        self.session_id = None
        
    def invoke(self, prompt: str) -> str:
        import requests
        import json
        
        if not self.chat_id or not self.session_id:
            raise ValueError("请先设置chat_id和session_id")
            
        logging.info(f"RAGFlow调用: 使用chat_id={self.chat_id}, session_id={self.session_id}")
        
        # 调用RAGFlow的chat completion接口
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            "question": prompt,
            "stream": False,
            "session_id": self.session_id
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/chats/{self.chat_id}/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0 and "data" in result:
                return result["data"]["answer"]
            else:
                logging.error(f"RAGFlow API error: {result}")
                return ""
                
        except Exception as e:
            logging.error(f"Error calling RAGFlow API: {str(e)}")
            return ""
            
    def list_chat_assistants(self) -> list:
        """获取所有聊天助手列表"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            response = requests.get(
                f"{self.base_url}/api/v1/chats",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0 and "data" in result:
                return result["data"]
            return []
            
        except Exception as e:
            logging.error(f"Error listing chat assistants: {str(e)}")
            return []
            
    def list_sessions(self, chat_id: str) -> list:
        """获取指定聊天助手的会话列表"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            response = requests.get(
                f"{self.base_url}/api/v1/chats/{chat_id}/sessions",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0 and "data" in result:
                return result["data"]
            return []
            
        except Exception as e:
            logging.error(f"Error listing sessions: {str(e)}")
            return []
            
    def create_session(self, chat_id: str, name: str = "New Session") -> Optional[str]:
        """创建新的会话"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            data = {
                "name": name
            }
            response = requests.post(
                f"{self.base_url}/api/v1/chats/{chat_id}/sessions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") == 0 and "data" in result:
                return result["data"].get("id")
            return None
            
        except Exception as e:
            logging.error(f"Error creating session: {str(e)}")
            return None

    def set_chat_session(self, chat_id: str, session_id: str):
        """设置当前使用的chat_id和session_id"""
        self.chat_id = chat_id
        self.session_id = session_id
        logging.info(f"RAGFlow设置会话: chat_id={chat_id}, session_id={session_id}")

def create_llm_adapter(
    interface_format: str,
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int
) -> BaseLLMAdapter:
    """
    工厂函数：根据 interface_format 返回不同的适配器实例。
    """
    if interface_format.lower() == "deepseek":
        return DeepSeekAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif interface_format.lower() == "openai":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif interface_format.lower() == "ollama":
        return OllamaAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif interface_format.lower() == "ml studio":
        return MLStudioAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif interface_format.lower() == "ragflow":
        return RAGFlowAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    else:
        raise ValueError(f"Unknown interface_format: {interface_format}")
