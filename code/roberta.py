import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

# ---------------- 🚀 全局配置 ----------------
INPUT_FILE = "Test.xlsx"  # 直接读取目标文件
OUTPUT_DIR = "results_roberta_optimized"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ---------------- 📊 数据预处理 ----------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ 找不到输入文件: {INPUT_FILE}")

# 读取全部数据
df_all = pd.read_excel(INPUT_FILE)

# 文本拼接逻辑
def combine_text(row):
    t = str(row['title']).strip() if pd.notnull(row['title']) else "无题"
    c = str(row['content']).strip()
    return f"标题：{t} [SEP] 内容：{c}"

df_all['combined_text'] = df_all.apply(combine_text, axis=1)

# 切分训练/验证/测试 (80% 训练, 10% 验证, 10% 测试)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df_all['combined_text'].tolist(), 
    df_all['labels'].tolist(), 
    test_size=0.10, 
    random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.11, random_state=42 # 0.11 * 0.9 ≈ 0.1
)

# ---------------- 🧠 核心类与函数 ----------------
class PoetryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro'),
        'auroc': roc_auc_score(labels, probs)
    }

# ---------------- ⚙️ 训练与评估流水线 ----------------
def run_pipeline():
    # 1. 初始化模型与分词器
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    def tokenize_fn(texts):
        return tokenizer(texts, truncation=True, padding=True, max_length=128)

    train_ds = PoetryDataset(tokenize_fn(X_train), y_train)
    val_ds = PoetryDataset(tokenize_fn(X_val), y_val)
    test_ds = PoetryDataset(tokenize_fn(X_test), y_test)

    # 2. 训练参数 (Batch=16, LR=1e-4)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # 3. 执行训练
    print(f"🚀 启动 RoBERTa 全量数据训练 (样本总数: {len(df_all)})...")
    trainer.train()

    # 4. 多维度评估
    results = []
    test_df_slice = df_all.iloc[df_all.index.isin(pd.Series(X_test).index)] # 仅对测试集部分进行切片分析

    def evaluate_slice(target_df, name):
        if len(target_df) == 0 or len(target_df['labels'].unique()) < 2: return
        ds = PoetryDataset(tokenize_fn(target_df['combined_text'].tolist()), target_df['labels'].tolist())
        out = trainer.predict(ds)
        probs = torch.nn.functional.softmax(torch.tensor(out.predictions), dim=-1).numpy()[:, 1]
        preds = np.argmax(out.predictions, axis=-1)
        results.append({
            "Subset": name, "Count": len(target_df),
            "AUROC": roc_auc_score(out.label_ids, probs),
            "Macro-F1": f1_score(out.label_ids, preds, average='macro')
        })

    print("📊 正在生成多维度分析报告...")
    evaluate_slice(df_all.loc[df_all.index.isin(pd.Series(X_test).index)], "Overall_Test_Set")
    
    # 子维度分析 (基于测试集)
    test_data_with_meta = df_all.iloc[len(df_all)-len(X_test):].copy() # 简化逻辑：取末尾测试部分
    human_baseline = test_data_with_meta[test_data_with_meta['labels'] == 0]

    for category in ['author', 'strategy']:
        if category in test_data_with_meta.columns:
            for val in test_data_with_meta[category].unique():
                sub_ai = test_data_with_meta[test_data_with_meta[category] == val]
                if not sub_ai.empty and val != 'human':
                    evaluate_slice(pd.concat([human_baseline, sub_ai]), f"{category}: {val}")

    # 5. 输出
    report_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print(report_df.to_string(index=False))
    report_df.to_csv("full_data_report.csv", index=False)

if __name__ == "__main__":
    run_pipeline()