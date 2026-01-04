import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- 🚀 配置区域 ----------------
MODEL_PATH = "Qwen/Qwen2.5-3B" 
INPUT_FILE = "Text.xlsx"
TEST_SIZE = 0.2  # 20% 作为测试集

# ---------------- 🧠 核心逻辑 ----------------

def get_qwen_features(texts, model, tokenizer):
    features = []
    batch_size = 2 
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Features"):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1536, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            # 提取最后一层最后一个 Token 的隐藏状态
            last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32).cpu().numpy()
            features.append(last_hidden)
            if i % 10 == 0: torch.cuda.empty_cache()
    return np.vstack(features)

def prepare_grouped_data(df, sz):
    texts, labels = [], []
    for lab in [0, 1]:
        sub = df[df['labels'] == lab]
        for i in range(0, len(sub), sz):
            chunk = sub.iloc[i : i + sz]
            if len(chunk) == sz:
                # 使用 Qwen 专用的结束符拼接多篇
                concat = " <|endoftext|> ".join([f"标题：{r.title} 内容：{r.content}" for _, r in chunk.iterrows()])
                texts.append(concat)
                labels.append(lab)
    return texts, np.array(labels)

def run_experiment():
    # 1. 加载模型（双卡自动分布）
    print("🤖 正在加载 Qwen2.5-3B 并分布至双卡...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # 2. 读取并分割数据
    if not os.path.exists(INPUT_FILE): raise FileNotFoundError(f"找不到文件: {INPUT_FILE}")
    df_full = pd.read_excel(INPUT_FILE)
    
    # 正常随机分割
    train_df, test_df = train_test_split(df_full, test_size=TEST_SIZE, random_state=42)
    print(f"✅ 数据加载完成。总样本: {len(df_full)}, 训练集: {len(train_df)}, 测试集: {len(test_df)}")

    final_results = []

    # 3. 实验循环
    for sz in [6, 12]:
        print(f"\n🧪 正在执行 {sz} 篇拼接实验...")
        train_txt, y_train = prepare_grouped_data(train_df, sz)
        test_txt, y_test = prepare_grouped_data(test_df, sz)

        X_train = get_qwen_features(train_txt, model, tokenizer)
        X_test = get_qwen_features(test_txt, model, tokenizer)

        # 补全三种分类器
        clfs = {
            "LL (Linear)": LogisticRegression(penalty=None, max_iter=2000),
            "LR (LogReg)": LogisticRegression(penalty='l2', C=1.0, max_iter=2000),
            "LRR (Ridge)": RidgeClassifier(alpha=1.0)
        }

        for name, clf in clfs.items():
            clf.fit(X_train, y_train)
            
            # 获取概率或决策分值
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                y_prob = clf.decision_function(X_test)
            
            # 判定预测值 (概率以0.5为界，Ridge以0为界)
            y_pred = (y_prob > (0.5 if hasattr(clf, "predict_proba") else 0)).astype(int)
            
            final_results.append({
                "GroupSize": sz,
                "Method": name,
                "AUROC": roc_auc_score(y_test, y_prob),
                "F1": f1_score(y_test, y_pred, average='macro')
            })

    # 4. 输出报告
    report = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("📊 Qwen2.5 特征提取鉴定实验报告 (三分类器)")
    print("="*70)
    print(report.to_string(index=False))
    report.to_csv("qwen_full_report.csv", index=False)


run_experiment()



import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# 1. 加载单首诗词的结果文件
df = pd.read_excel("Test_Qwen_Metrics_With_Title.xlsx")

def evaluate_best_macro_f1(y_true, y_scores, name):
    best_macro_f1 = 0
    best_threshold = 0

    auc = roc_auc_score(y_true, y_scores)
    if auc < 0.5:
        auc = 1 - auc

    thresholds = np.percentile(y_scores, np.linspace(0, 100, 200))

    for t in thresholds:
        # 尝试正向判定
        y_pred = (y_scores > t).astype(int)
        f1_pos = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 尝试反向判定
        y_pred_rev = (y_scores < t).astype(int)
        f1_neg = f1_score(y_true, y_pred_rev, average='macro', zero_division=0)

        current_f1 = max(f1_pos, f1_neg)

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_threshold = t

    y_pred_final = (y_scores > best_threshold).astype(int)
    if f1_score(y_true, y_pred_final, average='macro', zero_division=0) < best_macro_f1:
        y_pred_final = (y_scores < best_threshold).astype(int)

    precision_mac = precision_score(y_true, y_pred_final, average='macro', zero_division=0)
    recall_mac = recall_score(y_true, y_pred_final, average='macro', zero_division=0)

    # --- 4. 打印结果 ---
    print(f"--- 指标: {name} (Macro Evaluation) ---")
    print(f"AUROC:          {auc:.4f}")
    print(f"Macro-F1 Score: {best_macro_f1:.4f}")
    print(f"Macro-Precision:{precision_mac:.4f}")
    print(f"Macro-Recall:   {recall_mac:.4f}")
    print(f"最优阈值:       {best_threshold:.4f}\n")

# 运行评估
y_true = df['labels']

for col in ['ll', 'log_rank', 'lrr']:
    if col in df.columns:
        evaluate_best_macro_f1(y_true, df[col], col)