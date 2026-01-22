INSTRUCTION_FOR_AGENTS="""
# 变量和数据读取：
在addtional_args中，你会收到一个json,key是变量名，value是文件路径。key可以作为变量直接调用。
示例：{"example_variable": "path/to/example.json"}
你的代码必须与如下的示例一致：
from pathlib import Path
import json # 这两步的导入是必须的，使用open打开文件会失败。
text = Path(example_variable).read_text(encoding='utf-8')
real_example_variable = json.loads(text) 
... 这里是你的中间处理逻辑。最终要返回给用户的变量假设为your_answer_variable。
final_answer(your_answer_variable) # 必须使用final_answer函数来返回结果，否则结果不能被返回。
# final_answer无需定义和导入，为内置函数。"""