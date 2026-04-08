from pathlib import Path
from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
import os
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import json


from utils import logger
from utils.prompts import get_instruction_for_agents, SIMPLE_AGENT_SYSTEM_PROMPT, SIMPLE_AGENT_DEBUG_TEMPLATE, get_simple_agent_var_instruction
from utils.temp_file import get_var_storage_info, save_variable_to_temp
from utils.helper import extract_code_from_response, build_variable_preamble
from llm import OpenAILikeLLM, LLMConfig
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


# ============================================================
# SimpleCodeAgent - 自实现的 Code Agent，不依赖 smolagents
# ============================================================


class SimpleCodeAgent:
    """
    自实现的 Code Agent，不依赖 smolagents。
    
    工作原理：
    1. 将任务描述发送给 LLM，要求它生成 Python 代码（用 <code></code> 包裹）
    2. 提取代码块，保存为临时 .py 文件并用当前 Python 环境执行
    3. 如果执行成功，返回 stdout 输出作为结果
    4. 如果执行失败，将错误信息反馈给 LLM 进行 debug，重新生成代码
    5. 重复直到成功或达到 max_steps 次数限制
    
    参数:
        model: 模型名称
        api_base: API 基础 URL
        api_key: API 密钥
        additional_authorized_imports: 允许使用的额外 Python 库（仅做提示，不做强制限制）
        **kwargs: 传递给 LLMConfig 的额外参数 (temperature, top_p, seed, max_tokens 等)
    """

    def __init__(
        self,
        model: str = "",
        api_base: str = os.getenv("API_BASE_DEFAULT", ""),
        api_key: str = os.getenv("API_KEY_DEFAULT", ""),
        additional_authorized_imports: List[str] = [],
        **kwargs,
    ):
        config = LLMConfig(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_tokens', None),
            seed=kwargs.get('seed', 42),
        )
        self.llm = OpenAILikeLLM(config=config)
        self.llm.set_system_prompt(SIMPLE_AGENT_SYSTEM_PROMPT)
        self.imports = additional_authorized_imports
        self.execution_timeout = kwargs.get('execution_timeout', 60)

    def run(
        self,
        input: str,
        max_steps: int = 3,
        additional_args: Dict[str, Any] = {},
    ) -> Optional[str]:
        """
        执行任务。
        
        参数:
            input: 用户的任务描述
            max_steps: 最大执行/重试步数（含首次执行）
            additional_args: 传递给代码的变量字典
            
        返回:
            执行成功时返回 stdout 输出字符串；失败返回 None
        """
        # 准备变量临时文件
        file_paths = {}
        var_paths = {}
        var_type_info = {}

        try:
            for key, value in additional_args.items():
                suffix, type_name = get_var_storage_info(value)
                temp_path = save_variable_to_temp(key, value, suffix, type_name)
                logger.info(f"[SimpleCodeAgent] 变量 '{key}' 保存到临时文件: {temp_path}")
                file_paths[key] = temp_path
                var_paths[key] = temp_path
                var_type_info[key] = type_name

            # 构建变量读取指令（复用已有的工具函数）
            var_instruction = get_simple_agent_var_instruction(var_type_info)
            full_query = var_instruction + input

            # 清空对话历史，开始新会话
            self.llm.clear_history()

            result = self._run_loop(full_query, var_paths, max_steps)
            return result

        except Exception as e:
            logger.error(f"[SimpleCodeAgent] 运行出错: {e}")
            return None
        finally:
            # 清理临时文件
            for temp_path in file_paths.values():
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"[SimpleCodeAgent] 无法删除临时文件 {temp_path}: {e}")

    def _run_loop(
        self,
        query: str,
        var_paths: Dict[str, str],
        max_steps: int,
    ) -> Optional[str]:
        """核心执行循环：生成代码 → 执行 → 成功则返回 / 失败则 debug 重试。"""
        
        separator = "=" * 60

        # 第 1 步：让 LLM 根据 query 生成代码
        logger.info(f"\n{separator}")
        logger.info(f"[SimpleCodeAgent] Step 1/{max_steps}: 请求 LLM 生成代码")
        logger.info(separator)
        logger.debug(f"[SimpleCodeAgent] 发送给 LLM 的 Prompt:\n{query}")
        logger.info(f"[SimpleCodeAgent] Prompt 长度: {len(query)} 字符")

        response = self.llm.chat(query, keep_history=True)
        code = extract_code_from_response(response.content)

        if code is None:
            logger.error(f"\n{separator}")
            logger.error("[SimpleCodeAgent] LLM 未返回有效的 <code></code> 代码块")
            logger.error(separator)
            logger.debug(f"[SimpleCodeAgent] LLM 原始回复:\n{response.content}")
            return None

        for step in range(1, max_steps + 1):
            logger.info(f"\n{separator}")
            logger.info(f"[SimpleCodeAgent] Step {step}/{max_steps}: 执行代码")
            logger.info(separator)
            logger.info(f"[SimpleCodeAgent] 生成的代码:\n{'-' * 40}\n{code}\n{'-' * 40}")

            success, output = self._execute_code(code, var_paths)

            if success:
                logger.info(f"\n{separator}")
                logger.info(f"[SimpleCodeAgent] ✓ 代码执行成功 (Step {step}/{max_steps})")
                logger.info(separator)
                output_preview = output.strip()[:500]
                logger.info(f"[SimpleCodeAgent] 执行结果:\n{output_preview}")
                if len(output.strip()) > 500:
                    logger.info(f"[SimpleCodeAgent] ... (结果已截断，总长 {len(output.strip())} 字符)")
                return output.strip() if output else ""

            # 执行失败，记录错误
            logger.warning(f"\n{separator}")
            logger.warning(f"[SimpleCodeAgent] ✗ Step {step}/{max_steps} 代码执行失败")
            logger.warning(separator)
            logger.warning(f"[SimpleCodeAgent] 错误信息:\n{output[:1000]}")

            # 如果已达到最大步数，不再重试
            if step >= max_steps:
                logger.error(f"\n{separator}")
                logger.error(f"[SimpleCodeAgent] 已达到最大步数 {max_steps}，停止重试")
                logger.error(separator)
                return None

            # 让 LLM debug 并生成新代码
            logger.info(f"\n{separator}")
            logger.info(f"[SimpleCodeAgent] Step {step}→{step+1}: 请求 LLM 修复代码")
            logger.info(separator)
            debug_msg = SIMPLE_AGENT_DEBUG_TEMPLATE.format(code=code, error=output)
            logger.debug(f"[SimpleCodeAgent] Debug Prompt:\n{debug_msg}")
            
            response = self.llm.chat(debug_msg, keep_history=True)
            new_code = extract_code_from_response(response.content)

            if new_code is None:
                logger.error("[SimpleCodeAgent] LLM debug 后未返回有效代码块")
                logger.debug(f"[SimpleCodeAgent] LLM debug 回复:\n{response.content}")
                return None

            code = new_code

        return None

    # final_answer 函数注入代码：让 LLM 生成的 final_answer() 调用等价于 print()
    _FINAL_ANSWER_SHIM = (
        "def final_answer(result):\n"
        "    \"\"\"内置函数：返回最终结果。\"\"\"\n"
        "    print(result)\n"
    )

    def _execute_code(
        self, code: str, var_paths: Dict[str, str]
    ) -> Tuple[bool, str]:
        """将代码保存到临时 .py 文件并用当前 Python 环境执行。
        
        返回:
            (success: bool, output: str) - 成功时 output 为 stdout，失败时为 stderr
        """
        # 在代码顶部注入 final_answer shim + 变量路径赋值
        preamble_parts = [self._FINAL_ANSWER_SHIM]
        var_preamble = build_variable_preamble(var_paths)
        if var_preamble:
            preamble_parts.append(var_preamble)
        preamble = "\n".join(preamble_parts)
        full_code = preamble + "\n\n" + code

        # 写入临时 .py 文件
        temp_fd, temp_script = tempfile.mkstemp(suffix='.py', prefix='simple_agent_')
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(full_code)

            # 使用当前 Python 解释器执行
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                cwd=os.getcwd(),
            )

            if result.returncode == 0:
                return True, result.stdout
            else:
                # 合并 stderr 和 stdout（有些错误信息可能在 stdout 里）
                error_output = result.stderr
                if result.stdout:
                    error_output = f"stdout:\n{result.stdout}\nstderr:\n{error_output}"
                return False, error_output

        except subprocess.TimeoutExpired:
            return False, f"代码执行超时（超过 {self.execution_timeout} 秒）"
        except Exception as e:
            return False, f"执行代码时出现异常: {type(e).__name__}: {e}"
        finally:
            try:
                os.remove(temp_script)
            except OSError:
                pass


def create_code_agent(
    model: str,
    api_base: str = os.getenv("API_BASE_DEFAULT"),
    api_key: str = os.getenv("API_KEY_DEFAULT"),
    tools: List = [],
    additional_authorized_imports: List[str] = [],
    **kwargs,
):
    """
    工厂函数：根据环境变量 USE_SIMPLE_CODE_AGENT 决定创建哪种 Agent。
    
    当 USE_SIMPLE_CODE_AGENT=True 时，使用自实现的 SimpleCodeAgent（不依赖 smolagents）；
    否则使用基于 smolagents 的 MyCodeAgent。
    
    参数与 MyCodeAgent 完全一致，SimpleCodeAgent 会自动忽略不支持的参数（如 tools）。
    """
    use_simple = os.getenv("USE_SIMPLE_CODE_AGENT", "false").strip().lower() in ("true", "1", "yes")

    if use_simple:
        logger.info("[create_code_agent] 使用 SimpleCodeAgent（自实现，不依赖 smolagents）")
        # SimpleCodeAgent 不支持 tools 和 max_print_outputs_length，过滤掉
        kwargs.pop("max_print_outputs_length", None)
        return SimpleCodeAgent(
            model=model,
            api_base=api_base,
            api_key=api_key,
            additional_authorized_imports=additional_authorized_imports,
            **kwargs,
        )
    else:
        logger.info("[create_code_agent] 使用 MyCodeAgent（基于 smolagents）")
        return MyCodeAgent(
            model=model,
            api_base=api_base,
            api_key=api_key,
            tools=tools,
            additional_authorized_imports=additional_authorized_imports,
            **kwargs,
        )


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

    # 示例：使用 SimpleCodeAgent（自实现，不依赖 smolagents）
    print("\n" + "=" * 60)
    print("测试 SimpleCodeAgent")
    print("=" * 60)
    simple_agent = SimpleCodeAgent(
        model=os.getenv("CODE_AGENT_MODEL_NAME", "siliconflow/Qwen/Qwen3-8B"),
        api_base=os.getenv("API_BASE_DEFAULT"),
        api_key=os.getenv("API_KEY_DEFAULT"),
        additional_authorized_imports=['pandas', 'numpy'],
    )
    simple_result = simple_agent.run(
        "按照以下方式计算，并返回最终结果：1. list中所有数字的累加求和; 2. 减去pandas dataframe的平均值; 3. 再减去numpy array的平均值",
        max_steps=3,
        additional_args={'integer_list':[i for i in range(1, 101)], 'pandas_dataframe': pd.DataFrame({'a': [1, 2, 3, 4, 5]}), 'numpy_array': np.array([1, 2, 3, 4, 5])}
    )
    print(f"SimpleCodeAgent result: {simple_result}") #应该是5050-3-3=5044