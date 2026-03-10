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

    'dataframe_pickle': '''# 读取Pandas DataFrame文件 (.pkl) - MultiIndex列
import pandas as pd
{var_name}_data = pd.read_pickle({var_name})''',
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
    
    # 添加通用的结束说明 + 一些 DataFrame 使用提示（特别是 MultiIndex 和布尔筛选）
    base_instruction += """
# 代码编写要求：
1. 按照上述示例读取变量数据。
2. 编写你的处理逻辑，并重点关注“最终分析结论”，而不是打印过程细节。
3. 对 Pandas DataFrame，注意以下规则：
   - 布尔筛选写成：filtro = df[df[列名] == 某个值]，不要写成 df[列名 == 某个值]。
   - 如果是 MultiIndex 列（多层表头），使用完整的列路径元组访问：
       例如: df[("企业培育（35分）", "招商引资落地项目（个）", "招商引资落地项目（个）排名")]。
   - 访问“区县名称”这一列时，也写成：
       filtro = df[df[("区县", "名称")] == "渝北区"]。
4. 严格限制 print / 日志输出：
   - 禁止print所有的中间变量，由于dataframe的所有尺寸信息已经全部提供给你，你不被允许再去生成任何调试测试类的分步代码
   - 你生成的代码应该简单扼要：基于输入的query，直接去dataframe里基于提供给你的数据结构去直接查询获得。
   - 代码的结构应该类似于：读取数据->生成类SQL逻辑的Python代码，去dataframe中查询需要的信息->基于获得的信息计算你需要的所有统计量，如均值，方差，排序等等->使用final_answer返回最终结果。
   - 多余的调试类代码内容不被允许。
5. 分析完成后，整理一段清晰的自然语言总结，并使用 final_answer(your_answer_variable) 返回最终结果字符串。
6. final_answer 是内置函数，无需定义或导入。
<SystemEnd>
<User>
"""
    
    return base_instruction