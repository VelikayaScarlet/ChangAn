import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# ---------------- 配置 ----------------
MODEL_PATH = "yuchuantian/AIGC_detector_zhv3"
INPUT_FILE = "Test.xlsx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_metrics(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    best_f1 = 0
    thresholds = np.percentile(y_scores, np.linspace(0, 100, 100))
    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return auc, best_f1

def run_experiment():
    # 1. 加载数据
    df_raw = pd.read_excel(INPUT_FILE)
    
    def combine_text(row):
        t = str(row['title']).strip() if pd.notnull(row['title']) else "无题"
        c = str(row['content']).strip()
        return f"标题：{t} [SEP] 内容：{c}"
    
    df_raw['combined_text'] = df_raw.apply(combine_text, axis=1)
    print(f"✅ 数据载入完成，样本量: {len(df_raw)}")

    # 2. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.float16
    ).eval()

    final_report = []

    # 3. 实验循环
    for sz in [1, 6, 12]:
        print(f"\n🚀 测试 GroupSize = {sz}...")
        all_probs = []
        all_labels = []
        all_authors = []
        all_strategies = []

        for lab in [0, 1]:
            sub_df = df_raw[df_raw['labels'] == lab]
            for i in tqdm(range(0, len(sub_df), sz), desc=f"Size-{sz}-Lab-{lab}"):
                chunk = sub_df.iloc[i : i + sz]
                if len(chunk) == sz:
                    text = " [SEP] ".join(chunk['combined_text'].tolist())
                    
                    with torch.no_grad():
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                        logits = model(**inputs).logits
                        prob = torch.softmax(logits, dim=-1)[0, 1].item()
                    
                    all_probs.append(prob)
                    all_labels.append(lab)
                    all_authors.append(chunk['author'].iloc[0] if 'author' in chunk.columns else "N/A")
                    all_strategies.append(chunk['strategy'].iloc[0] if 'strategy' in chunk.columns else "N/A")

        # 整体评估
        res_df = pd.DataFrame({
            'label': all_labels, 
            'score': all_probs, 
            'author': all_authors, 
            'strategy': all_strategies
        })
        
        auc, f1 = evaluate_metrics(res_df['label'], res_df['score'])
        final_report.append({"Size": sz, "Subset": "Overall", "AUROC": auc, "Macro-F1": f1})

        # 多维度细分评估 (仅针对 sz=1 进行详细拆解，或根据需求对所有 sz 拆解)
        if sz == 1:
            test_human = res_df[res_df['label'] == 0]
            # 作者维度
            for ai in ['Deepseek', 'kimi-k2', 'gpt-4.1', 'seed']:
                test_ai = res_df[res_df['author'] == ai]
                if not test_ai.empty:
                    sub = pd.concat([test_human, test_ai])
                    a, f = evaluate_metrics(sub['label'], sub['score'])
                    final_report.append({"Size": sz, "Subset": f"Author: {ai}", "AUROC": a, "Macro-F1": f})
            # 策略维度
            for strat in ['generate', 'critique']:
                test_strat = res_df[res_df['strategy'] == strat]
                if not test_strat.empty:
                    sub = pd.concat([test_human, test_strat])
                    a, f = evaluate_metrics(sub['label'], sub['score'])
                    final_report.append({"Size": sz, "Subset": f"Strategy: {strat}", "AUROC": a, "Macro-F1": f})

    # 4. 输出
    report_df = pd.DataFrame(final_report)
    print("\n" + "="*70)
    print(f"🏆 AIGC-Detector ({MODEL_PATH.split('/')[-1]}) 报告")
    print("="*70)
    print(report_df.to_string(index=False))
    report_df.to_excel("AIGC_Detector_V3_Full_Report.xlsx", index=False)

if __name__ == "__main__":
    run_experiment()