from pathlib import Path
from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple
import tempfile
import json


from utils import logger
from utils.prompts import get_instruction_for_agents
from utils.temp_file import get_var_storage_info, save_variable_to_temp
load_dotenv()


def _dump_messages_debug(messages, step_label=""):
    """将每一步发给 LLM 的 messages 的结构和大小记录到日志文件，方便排查 token 膨胀。
    摘要输出到终端 (info)，完整内容输出到日志文件 (debug)。
    """
    total_chars = 0
    details = []
    for i, msg in enumerate(messages):
        role = getattr(msg, 'role', msg.get('role', '?')) if isinstance(msg, dict) else getattr(msg, 'role', '?')
        if isinstance(msg, dict):
            content = msg.get('content', '')
        else:
            content = getattr(msg, 'content', '')
        if isinstance(content, list):
            text_parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
            text = '\n'.join(text_parts)
        else:
            text = str(content) if content else ''
        char_len = len(text)
        total_chars += char_len
        details.append((i, str(role), char_len, text))

    # 终端摘要
    summary_parts = [f"msg[{i}] role={role} len={clen}" for i, role, clen, _ in details]
    logger.info(f"[DEBUG] {step_label}: {len(messages)} 条消息, 总计 {total_chars} 字符 | {'; '.join(summary_parts)}")

    # 日志文件写完整内容
    logger.debug(f"===== {step_label} Messages 发送给 LLM =====")
    for i, role, char_len, text in details:
        if char_len > 600:
            preview = text[:300] + f"\n... [省略 {char_len - 600} 字符] ...\n" + text[-300:]
        else:
            preview = text
        logger.debug(f"  msg[{i}] role={role}, 长度={char_len} 字符\n{preview}")
    logger.debug(f"===== 总计 {len(messages)} 条消息, {total_chars} 字符 =====\n")




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
            max_print_outputs_length=kwargs.get('max_print_outputs_length', 2000),
        )
        self.imports = additional_authorized_imports
    
    def run(self, input: str, max_steps: int = 3, additional_args: Dict[str, Any] = {}) -> str:
        
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
                logger.info(f"temp_path: {temp_path}")
                file_paths[key] = temp_path
                processed_args[key] = temp_path
                var_type_info[key] = type_name
            
            try:
                # 生成包含读取示例的指令
                input = get_instruction_for_agents(var_type_info) + input
                logger.info(f"input: {input}")

                # === DEBUG: 拦截 LLM 调用，记录每步发送的完整 messages ===
                _original_generate = self.model.generate
                _step_counter = [0]

                def _debug_generate(messages, **kwargs):
                    _step_counter[0] += 1
                    _dump_messages_debug(messages, step_label=f"Step {_step_counter[0]}")
                    return _original_generate(messages, **kwargs)

                self.model.generate = _debug_generate

                final_output = self.main_agent.run(
                    input, 
                    additional_args=processed_args, 
                )
            except Exception as e:
                logger.error(f"Agent RuntimeError: {e}")
                return None
            finally:
                # 清理临时文件
                for temp_path in file_paths.values():
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Warning: 无法删除临时文件 {temp_path}: {e}")
            
            return final_output
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
if __name__ == "__main__":
    from importlib.metadata import version
    import pandas as pd
    import numpy as np
    print(f"LiteLLM version: {version('litellm')}")
    print(f"Smolagents version: {version('smolagents')}")

    # 示例：使用 CodeAgent 独立的模型环境变量
    agent = MyCodeAgent(
        model=os.getenv("CODE_AGENT_MODEL_NAME", "siliconflow/Qwen/Qwen3-8B"),
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