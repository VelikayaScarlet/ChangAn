import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os

# --- 1. 配置与模型加载 (8-bit 双卡分布) ---
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
s_model_id = "Qwen/Qwen2.5-7B"
r_model_id = "Qwen/Qwen2.5-3B"

print("正在并行加载模型...")
s_tokenizer = AutoTokenizer.from_pretrained(s_model_id)
s_model = AutoModelForCausalLM.from_pretrained(
    s_model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
)

r_tokenizer = AutoTokenizer.from_pretrained(r_model_id)
r_model = AutoModelForCausalLM.from_pretrained(
    r_model_id, quantization_config=bnb_config, device_map={"": 1}, trust_remote_code=True
)
print("✅ 模型加载完成！")

# --- 2. 核心检测算法 ---
def get_fast_detect_score(text, max_len=1024):
    with torch.no_grad():
        # GPU 0: 评分模型
        s_inputs = s_tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).to(0)
        s_logits = s_model(s_inputs.input_ids).logits
        s_ll = -F.cross_entropy(s_logits[:, :-1, :].reshape(-1, s_logits.size(-1)), 
                                s_inputs.input_ids[:, 1:].reshape(-1), reduction='mean')
        
        # GPU 1: 参考模型
        r_inputs = r_tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).to(1)
        r_logits = r_model(r_inputs.input_ids).logits
        r_ll = -F.cross_entropy(r_logits[:, :-1, :].reshape(-1, r_logits.size(-1)), 
                                r_inputs.input_ids[:, 1:].reshape(-1), reduction='mean')
        
        return (s_ll.cpu() - r_ll.cpu()).item()

# --- 3. 评测执行函数 ---
def evaluate_dataset(df, name, sz=1):
    if df.empty: return None
    
    y_true, y_scores = [], []
    
    # 根据 sz 决定是否需要拼接文本
    if sz == 1:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {name}"):
            text = f"标题：{row['title']} 内容：{row['content']}"
            y_scores.append(get_fast_detect_score(text))
            y_true.append(row['labels'])
    else:
        # 拼接评测逻辑：分别处理人类和 AI 块
        for lab in [0, 1]:
            sub = df[df['labels'] == lab]
            for i in range(0, len(sub) - sz + 1, sz):
                chunk = sub.iloc[i : i + sz]
                combined_text = " ".join([f"标题：{r.title} 内容：{r.content}" for _, r in chunk.iterrows()])
                y_scores.append(get_fast_detect_score(combined_text))
                y_true.append(lab)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = [1 if s > 0 else 0 for s in y_scores]
    
    return {
        "Subset": name,
        "Size": sz,
        "AUROC": roc_auc_score(y_true, y_scores),
        "Macro-F1": f1_score(y_true, y_preds, average='macro', zero_division=0),
        "Acc": accuracy_score(y_true, y_preds),
        "Count": len(y_true)
    }

# --- 4. 运行全量实验 ---
INPUT_FILE = "/kaggle/input/poetry-test-wlabels/Test.xlsx"
df_all = pd.read_excel(INPUT_FILE)
print(f"数据载入完成，总样本数: {len(df_all)}")

final_results = []

# A. 基础 Size=1 全量评测及细分评估
print("\n--- 正在执行单篇评测 ---")
final_results.append(evaluate_dataset(df_all, "Overall_Full", sz=1))

test_human = df_all[df_all['labels'] == 0]
for ai in ['Deepseek', 'kimi-k2', 'gpt-4.1', 'seed']:
    sub = df_all[df_all['author'] == ai]
    if not sub.empty:
        final_results.append(evaluate_dataset(pd.concat([test_human, sub]), f"Author: {ai}", sz=1))

for strat in ['generate', 'critique']:
    sub = df_all[df_all['strategy'] == strat]
    if not sub.empty:
        final_results.append(evaluate_dataset(pd.concat([test_human, sub]), f"Strategy: {strat}", sz=1))

# B. 拼接评测 Size=6, 12
print("\n--- 正在执行拼接评测 ---")
for bs in [6, 12]:
    final_results.append(evaluate_dataset(df_all, f"Mixed_Batch", sz=bs))

# --- 5. 结果输出 ---
report_df = pd.DataFrame([r for r in final_results if r is not None])
print("\n" + "="*70)
print("📊 Fast-DetectGPT (Qwen2.5 Dual GPU) 全量评测报告")
print("="*70)
print(report_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

report_df.to_excel("FastDetectGPT_Full_Report.xlsx", index=False)