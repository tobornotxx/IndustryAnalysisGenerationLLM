from pathlib import Path
from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
# import pandas as pd
from utils.prompts import INSTRUCTION_FOR_AGENTS
load_dotenv()   

class MyCodeAgent:
    """
    自包装CodeAgent
    
    参数:
        model: 模型名称或标识符（例如 "gpt-4", "claude-3-opus" 等）
        api_base: API 基础 URL，默认从环境变量 API_BASE_DEFAULT 读取
        api_key: API 密钥，默认从环境变量 API_KEY_DEFAULT 读取
        tools: 工具列表，默认 []，如果有，必须是使用smolagents @tool装饰器装饰。
        additional_authorized_imports: 额外授权导入的库列表，默认 []. 必须是Python库的名称。
        **llm_kwargs: 传递给 LiteLLMModel 的额外参数
    """
    def __init__(
        self, 
        model: str,
        api_base: str = os.getenv("API_BASE_DEFAULT"),
        api_key: str = os.getenv("API_KEY_DEFAULT"),
        tools: List[tool] = [],
        additional_authorized_imports: List[str] = [],
        **kwargs,
    ):
        temperature = kwargs.get('temperature', 1.0)
        top_p = kwargs.get('top_p', 1.0)
        seed = kwargs.get('seed', 42)
        self.model = LiteLLMModel(
            model_id=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            top_p = top_p,
            seed = seed,
        )
        # 默认授权导入pathlib和json
        if 'pathlib' not in additional_authorized_imports:
            additional_authorized_imports.append('pathlib')
        if 'json' not in additional_authorized_imports:
            additional_authorized_imports.append('json')

        self.main_agent = CodeAgent(
            model=self.model,
            tools=tools,
            additional_authorized_imports=additional_authorized_imports,
        )
    
    def run(self, input: str, max_steps: int = 10, additional_args: Dict[str, Any] = {}) -> str:
        
        try:
            import tempfile
            import json
            # 使用临时文件+文件路径的方式传入变量，节省上下文。
            file_paths = {}  
            processed_args = {}
            
            for key, value in additional_args.items():
                # 创建临时文件
                temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix=f'{key}_', text=True)
                file_paths[key] = temp_path
                # 将值保存为 JSON 格式
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                
                # 将值替换为文件路径
                processed_args[key] = temp_path
            try:
                input += INSTRUCTION_FOR_AGENTS
                final_output = self.main_agent.run(
                    input, 
                    additional_args=processed_args, 
                )
            except Exception as e:
                print(f"Agent RuntimeError: {e}")
                return None
            finally:
                # 清理临时文件
                for temp_path in file_paths.values():
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Warning: 无法删除临时文件 {temp_path}: {e}")
            
            return final_output
            # Future Update: 决定好传入的文件类型。prompt灵活组织，指导llm调哪个库。注意不能with open，不支持。
        except Exception as e:
            print(f"Error: {e}")
            return None
if __name__ == "__main__":
    from importlib.metadata import version
    print(f"LiteLLM version: {version('litellm')}")
    print(f"Smolagents version: {version('smolagents')}")

    agent = MyCodeAgent(
        model='siliconflow/Qwen/Qwen3-8B',
        api_base=os.getenv("API_BASE_DEFAULT"),
        api_key=os.getenv("API_KEY_DEFAULT"),
        tools=[],
        additional_authorized_imports=[],
    )
    result = agent.run(
        "计算给定的list中所有数字的累加求和",
        additional_args={'integer_list':[i for i in range(1, 101)]}
    )
    print(f"result: {result}")