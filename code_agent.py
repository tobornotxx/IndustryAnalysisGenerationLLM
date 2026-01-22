from pathlib import Path
from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple
import tempfile
import json



from utils.prompts import get_instruction_for_agents
from utils.temp_file import get_var_storage_info, save_variable_to_temp
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
        self.imports = additional_authorized_imports
    
    def run(self, input: str, max_steps: int = 10, additional_args: Dict[str, Any] = {}) -> str:
        
        try:
            # 使用临时文件+文件路径的方式传入变量，节省上下文。
            file_paths = {}  # key -> temp_path (用于清理)
            processed_args = {}  # key -> temp_path (传给agent)
            var_type_info = {}  # key -> type_name (用于生成prompt)
            
            for key, value in additional_args.items():
                # 根据变量类型确定存储格式
                suffix, type_name = get_var_storage_info(value)
                
                # 保存变量到临时文件
                temp_path = save_variable_to_temp(key, value, suffix, type_name)
                
                file_paths[key] = temp_path
                processed_args[key] = temp_path
                var_type_info[key] = type_name
            
            try:
                # 生成包含读取示例的指令
                input = get_instruction_for_agents(var_type_info) + input
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
        except Exception as e:
            print(f"Error: {e}")
            return None
if __name__ == "__main__":
    from importlib.metadata import version
    import pandas as pd
    import numpy as np
    print(f"LiteLLM version: {version('litellm')}")
    print(f"Smolagents version: {version('smolagents')}")

    agent = MyCodeAgent(
        model='siliconflow/Qwen/Qwen3-8B',
        api_base=os.getenv("API_BASE_DEFAULT"),
        api_key=os.getenv("API_KEY_DEFAULT"),
        tools=[],
        additional_authorized_imports=['pandas', 'numpy'],
    )
    result = agent.run(
        "按照以下方式计算，并返回最终结果：1. list中所有数字的累加求和; 2. 减去pandas dataframe的平均值; 3. 再减去numpy array的平均值",
        additional_args={'integer_list':[i for i in range(1, 101)], 'pandas_dataframe': pd.DataFrame({'a': [1, 2, 3, 4, 5]}), 'numpy_array': np.array([1, 2, 3, 4, 5])}
    )
    print(f"result: {result}") #应该是5050-3-3=5044