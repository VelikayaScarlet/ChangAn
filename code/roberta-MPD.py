import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- 🚀 配置区域 ----------------
INPUT_FILE = "Test.xlsx"
MODEL_PATH = "results_roberta/checkpoint" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_concatenated_test(group_sizes=[6, 12]):
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ 找不到文件: {INPUT_FILE}")
    
    df_all = pd.read_excel(INPUT_FILE)
    
    def combine_text(row):
        t = str(row['title']).strip() if pd.notnull(row['title']) else "无题"
        c = str(row['content']).strip()
        return f"标题：{t}。内容：{c}"
    
    df_all['combined_text'] = df_all.apply(combine_text, axis=1)
    
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    overall_results = []

    for sz in group_sizes:
        print(f"\n🚀 正在执行 {sz} 篇拼接推理...")
        
        test_human = df_all[df_all['labels'] == 0]
        test_ai = df_all[df_all['labels'] == 1]
        
        all_group_texts = []
        all_group_labels = []

        for sub_df in [test_human, test_ai]:
            for i in range(0, len(sub_df), sz):
                chunk = sub_df.iloc[i : i + sz]
                if len(chunk) == sz:
                    concat_text = " [SEP] ".join(chunk['combined_text'].tolist())
                    all_group_texts.append(concat_text)
                    all_group_labels.append(chunk['labels'].iloc[0])

        all_probs = []
        # 恢复较大的 Batch Size 以利用并行性能
        batch_size = 64 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_group_texts), batch_size), desc=f"Size {sz}"):
                batch = all_group_texts[i : i + batch_size]
                inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
                outputs = model(**inputs)
                p = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(p)

        y_true = np.array(all_group_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob > 0.5).astype(int)
        
        overall_results.append({
            "Experiment": f"{sz}-Poem Concat",
            "Group_Count": len(y_true),
            "AUROC": roc_auc_score(y_true, y_prob),
            "Macro-F1": f1_score(y_true, y_pred, average='macro')
        })

    report_df = pd.DataFrame(overall_results)
    print("\n" + "="*70)
    print(report_df.to_string(index=False))
    report_df.to_csv("concatenation_report.csv", index=False)

if __name__ == "__main__":
    run_concatenated_test(group_sizes=[6, 12])