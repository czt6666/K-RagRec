"""
修复 LLaMA-2 BASE 模型使用 Chat 模板导致的 Collapse 问题。

根因：代码加载的是 Llama-2-7b-hf（BASE 模型），但 graph_llm.py 里硬编码了
      BOS='<s>[INST]' 和 EOS_USER='[/INST]'，这些是 Chat 模型的特殊 token。
      BASE 模型的 tokenizer 会把 [INST] 拆成普通字符，模型无法理解指令边界，
      导致输出 collapse 到固定字母。

修复方案（二选一）：
  A. 换用 Chat 模型（推荐，但需重新下载/有 gated 限制）
  B. 修改 Prompt 模板为纯文本格式（BASE 模型友好，立即可用）

本脚本采用方案 B，修改 graph_llm.py 中的模板和 token 拼接逻辑。

用法：
    python tools/fix_chat_template.py
"""
import shutil
import os

path = "methods/baseline/src/model/graph_llm.py"
backup = path + ".bak2"
shutil.copy(path, backup)

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 方案 B：把 Chat 格式模板改成 BASE 模型友好的纯文本格式
# 原模板（Chat 风格，BASE 模型无法理解）：
#   BOS = '<s>[INST]'
#   EOS_USER = '[/INST]'
#   EOS = '</s>'
#   Prompt: "Below is an instruction... ### Instruction: ... ###Options: ..."
#
# 新模板（BASE 模型友好）：
#   BOS = '<s>'
#   EOS_USER = '\nAnswer: '
#   EOS = '</s>'
#   Prompt 简化为直接的问题描述，去掉 [INST] 包装

old_bos = "BOS = '<s>[INST]'"
new_bos = "BOS = '<s>'"
if old_bos in content:
    content = content.replace(old_bos, new_bos)
    print("[FIXED] BOS: '<s>[INST]' -> '<s>'")

old_eos_user = "EOS_USER = '[/INST]'"
new_eos_user = "EOS_USER = '\\nAnswer: '"
if old_eos_user in content:
    content = content.replace(old_eos_user, new_eos_user)
    print("[FIXED] EOS_USER: '[/INST]' -> '\\nAnswer: '")

# 同时修改 train.py 里的 prompt 模板，确保 train/eval 一致
# 但 graph_llm.py 里的 question 是从 sample['question'] 来的，而 sample['question']
# 是在 train.py / evaluate.py 里构建的。所以主要改 train.py 和 evaluate.py 的 prompt。

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

# 修改 train.py 的 prompt 模板
train_path = "methods/baseline/train.py"
shutil.copy(train_path, train_path + ".bak2")
with open(train_path, 'r', encoding='utf-8') as f:
    train_content = f.read()

# 把 Chat 风格的 Prompt 改成 BASE 模型友好的格式
# 原: "Below is an instruction ... ### Instruction: Given... ###Options: ..."
# 新: 直接简洁的问题描述
old_train_prompt = '''query.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                            ### Instruction: Given the user's watching history, selecting a film that is most likely to interest the user from the options. ### Watching history: {input}. ###Options: {question}. Select a movie from options A to T that the user is most likely to be interested in. Please answer "A" or "B" or "C" or "D" or "E" or "F" or "G" or "H" or "I" or "J" or "K" or "L" or "M" or "N" or "O" or "P" or "Q" or "R" or "S" or "T" only\".""")'''

new_train_prompt = '''query.append(f"The user has watched the following movies: {input}. Based on this watching history, select the movie from the following options that the user is most likely to enjoy. Options: {question}. Answer with a single letter from A to T.")'''

if old_train_prompt in train_content:
    train_content = train_content.replace(old_train_prompt, new_train_prompt)
    print("[FIXED] train.py: Prompt 模板已改为 BASE 模型友好的纯文本格式")
else:
    print("[WARN] train.py: 没找到旧 prompt，可能位置或格式有变化，请手动检查")

with open(train_path, 'w', encoding='utf-8') as f:
    f.write(train_content)

# 修改 evaluate.py 的 prompt 模板（如果存在的话）
eval_path = "methods/baseline/evaluate.py"
if os.path.exists(eval_path):
    shutil.copy(eval_path, eval_path + ".bak2")
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_content = f.read()

    # evaluate.py 里通常也有类似的 prompt 构造
    if "### Instruction: Given the user's watching history" in eval_content:
        eval_content = eval_content.replace(
            "## Instruction: Given the user's watching history",
            "Given the user's watching history"
        )
        eval_content = eval_content.replace(
            "### Instruction: Given the user's watching history",
            "Given the user's watching history"
        )
        print("[FIXED] evaluate.py: prompt 模板已对齐")

    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write(eval_content)

print("\n完成。备份文件: .bak2")
print("\n⚠️  重要提示：")
print("  这个修复把 Prompt 从 Chat 格式改成了 BASE 模型友好的纯文本格式。")
print("  如果你服务器上有 Llama-2-7b-chat-hf，更推荐直接换模型而不是改模板。")
print("  修改后需要重新训练，因为 prompt 结构变了。")
