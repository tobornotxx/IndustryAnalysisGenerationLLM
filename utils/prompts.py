from typing import List, Dict


# 各种文件类型的读取代码示例
READ_EXAMPLES = {
    'txt': '''# 读取文本文件 (.txt) - 适用于 str, int, float, bool 类型
from pathlib import Path
text_content = Path({var_name}).read_text(encoding='utf-8')
# 如果原始数据是数值类型，需要转换：
# int_value = int(text_content)
# float_value = float(text_content)
# bool_value = text_content.lower() == 'true'
{var_name}_data = text_content''',

    'json': '''# 读取JSON文件 (.json) - 适用于 list, dict 类型
from pathlib import Path
import json
text = Path({var_name}).read_text(encoding='utf-8')
{var_name}_data = json.loads(text)''',

    'ndarray': '''# 读取NumPy数组文件 (.npy)
import numpy as np
{var_name}_data = np.load({var_name})''',

    'dataframe': '''# 读取Pandas DataFrame文件 (.parquet)
import pandas as pd
{var_name}_data = pd.read_parquet({var_name})''',
}


def get_instruction_for_agents(var_type_info: Dict[str, str] = None) -> str:
    """
    生成针对当前传入变量类型的读取指令
    
    参数:
        imports: 授权导入的库列表
        var_type_info: 变量名到类型名的映射 {var_name: type_name}
                       type_name 可以是: 'txt', 'json', 'ndarray', 'dataframe'
    """
    if var_type_info is None:
        var_type_info = {}
    
    # 基础说明
    base_instruction = """<SystemStart>
# 变量和数据读取：
在additional_args中，你会收到一个字典，key是变量名，value是文件路径。key可以作为变量直接调用。
重要提示：使用 open() 打开文件会失败，必须使用下面展示的读取方法！
"""
    
    # 如果没有传入变量，返回通用说明
    if not var_type_info:
        base_instruction += """
示例：{"example_variable": "path/to/example.json"}
```python
from pathlib import Path
import json
text = Path(example_variable).read_text(encoding='utf-8')
real_example_variable = json.loads(text)
```
"""
    else:
        # 根据传入变量类型生成具体的读取示例
        base_instruction += "\n当前传入的变量及其读取方法：\n"
        
        for var_name, type_name in var_type_info.items():
            example_code = READ_EXAMPLES.get(type_name, READ_EXAMPLES['json'])
            formatted_code = example_code.format(var_name=var_name)
            base_instruction += f"\n## 变量 `{var_name}` (类型: {type_name}):\n```python\n{formatted_code}\n```\n"
    
    # 添加通用的结束说明
    base_instruction += """
# 代码编写要求：
1. 按照上述示例读取变量数据
2. 编写你的处理逻辑
3. 必须使用 final_answer(your_answer_variable) 返回最终结果
4. final_answer 是内置函数，无需定义或导入
<SystemEnd>
<User>
"""
    
    return base_instruction