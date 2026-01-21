INSTRUCTION_FOR_AGENTS="""
You are provided some python variable names and corresponding file path. 
You do not have those variables directly, you must always use Path(path to that file).read_text(),files use utf8 as encode parameter. from that text, you should use json.loads() to get access to the original variable.
When task completed, always use final_answer(final variable) to return your code result to user."""