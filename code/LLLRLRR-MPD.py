import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

# ---------------- 🚀 1. 配置与模型加载 ----------------
MODEL_PATH = "Qwen/Qwen2.5-3B"
INPUT_FILE = "Test.xlsx"

print("🤖 正在加载 Qwen2.5-3B (双卡分布)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

def compute_intrinsic_metrics(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan, np.nan, np.nan

    # 1536 长度足以容纳 12 篇拼接后的诗词
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # --- LL (Log-Likelihood) 计算 ---
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()

    # 使用 CrossEntropy 计算每个 token 的负对数似然
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    nll_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ll = -nll_per_token.mean().item()

    # --- Log-Rank 性能优化版计算 ---
    # 优化点：不再使用全局 argsort，而是使用 torch.gather 计算目标 label 的 rank
    # 这种方式在 Vocab 很大时速度提升 10 倍以上
    token_logits = shift_logits.view(-1, shift_logits.size(-1))
    token_labels = shift_labels.view(-1, 1)
    
    # 计算比目标 label 的 logit 更高的数量，即为 rank
    ranks = (token_logits > torch.gather(token_logits, 1, token_labels)).sum(dim=-1)
    log_rank = torch.log(ranks.float() + 1).mean().item()

    # --- LRR (Log-Rank Ratio) ---
    lrr = ll / log_rank if log_rank != 0 else 0

    return ll, log_rank, lrr

# ---------------- 📊 2. 数据处理 (全量读取 + 分割) ----------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ 找不到文件: {INPUT_FILE}")

raw_df = pd.read_excel(INPUT_FILE)
# 正常分割数据，保证实验的科学性
train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
print(f"✅ 全量数据载入。测试集规模: {len(test_df)} 条")

# ---------------- 🧪 3. 执行指标提取 ----------------
all_results = []

# 选取 6 篇和 12 篇进行实验
for sz in [6, 12]:
    print(f"\n🚀 正在计算 GroupSize={sz} 的内生指标...")
    
    # 我们通常只对测试集进行内生指标评估（Zero-shot 场景）
    for lab in [0, 1]:
        sub_df = test_df[test_df['labels'] == lab]
        
        for i in tqdm(range(0, len(sub_df), sz), desc=f"Label={lab}"):
            chunk = sub_df.iloc[i : i + sz]
            if len(chunk) == sz:
                # 拼接逻辑：标题1 内容1 标题2 内容2 ...
                combined_text = " ".join([f"{r['title']} {r['content']}" for _, r in chunk.iterrows()])
                
                ll, lrank, lrr = compute_intrinsic_metrics(combined_text)
                
                all_results.append({
                    "group_size": sz,
                    "labels": lab,
                    "ll": ll,
                    "log_rank": lrank,
                    "lrr": lrr
                })

# ---------------- 💾 4. 保存 ----------------
final_df = pd.DataFrame(all_results)
output_name = "Intrinsic_Metrics_Report_Full.xlsx"
final_df.to_excel(output_name, index=False)

print("\n" + "="*50)
print(f"✅ 计算完成！结果保存至: {output_name}")
print(f"📊 统计摘要 (按 GroupSize 分组):")
print(final_df.groupby(['group_size', 'labels'])[['ll', 'lrr']].mean())


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# 1. 加载数据
df = pd.read_excel("Intrinsic_Metrics_Report_Full.xlsx")

def evaluate_best_macro_f1(y_true, y_scores, name):
    """
    寻找最优阈值并计算 Macro-F1, Macro-Precision, Macro-Recall 和 AUROC
    """
    best_macro_f1 = 0
    best_threshold = 0
    
    # AUROC 本身不分 binary 或 macro，它衡量的是排序能力
    auc = roc_auc_score(y_true, y_scores)
    if auc < 0.5:
        auc = 1 - auc
    
    # 搜索阈值
    thresholds = np.percentile(y_scores, np.linspace(0, 100, 200))

    for t in thresholds:
        # 尝试正向判定 (average='macro')
        y_pred = (y_scores > t).astype(int)
        f1_mac_pos = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 尝试反向判定
        y_pred_rev = (y_scores < t).astype(int)
        f1_mac_neg = f1_score(y_true, y_pred_rev, average='macro', zero_division=0)
        
        current_f1 = max(f1_mac_pos, f1_mac_neg)
        
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_threshold = t

    # 使用最优阈值锁定最终结果
    y_pred_final = (y_scores > best_threshold).astype(int)
    if f1_score(y_true, y_pred_final, average='macro', zero_division=0) < best_macro_f1:
        y_pred_final = (y_scores < best_threshold).astype(int)

    # 汇报 Macro 平均的指标
    precision_mac = precision_score(y_true, y_pred_final, average='macro', zero_division=0)
    recall_mac = recall_score(y_true, y_pred_final, average='macro', zero_division=0)

    # 格式化打印结果
    print(f"| {name:<10} | {auc:.4f} | {best_macro_f1:.4f} | {precision_mac:.4f} | {recall_mac:.4f} |")

# 2. 运行评估
for sz in sorted(df['group_size'].unique()):
    print(f"\n" + "="*70)
    print(f"🚀 实验结果汇报 (Macro Metrics): GroupSize = {sz}")
    print("-" * 70)
    print(f"| {'指标':<10} | {'AUROC':<6} | {'Macro-F1':<6} | {'M-Prec':<6} | {'M-Rec':<6} |")
    print("-" * 70)
    
    group_df = df[df['group_size'] == sz]
    y_true = group_df['labels']
    
    for col in ['ll', 'log_rank', 'lrr']:
        evaluate_best_macro_f1(y_true, group_df[col], col)
    print("-" * 70)