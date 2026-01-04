import pandas as pd
from openai import OpenAI
import time
import os

# ---------------- 🚀 配置区域 ----------------
# 可选值: 'DEEPSEEK', 'KIMI', 'GPT', 'DOUBAO'
CHOSEN_MODEL = 'DEEPSEEK' 

# 模型配置字典 (请在此处填写你的 API Key)
CONFIGS = {
    'DEEPSEEK': {
        'api_key': 'YOUR_API_KEY',
        'base_url': "https://api.deepseek.com",
        'model': "deepseek-chat"
    },
    'KIMI': {
        'api_key': 'YOUR_API_KEY',
        'base_url': "https://api.moonshot.cn/v1",
        'model': "kimi-k2-turbo-preview"
    },
    'GPT': {
        'api_key': "YOUR_API_KEY",
        'base_url': "https://api.vectorengine.ai/v1",
        'model': "gpt-4.1"
    },
    'DOUBAO': {
        'api_key': "YOUR_API_KEY",
        'base_url': "https://ark.cn-beijing.volces.com/api/v3",
        'model': "ep-xxxxxx" 
    }
}

cfg = CONFIGS[CHOSEN_MODEL]
INPUT_FILE = 'Test_long.xlsx'
OUTPUT_FILE = f'Result_Batch_{CHOSEN_MODEL}.xlsx'
SUMMARY_FILE = f'Batch_Summary_{CHOSEN_MODEL}_6.xlsx'

client = OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])

SYSTEM_PROMPT = "你是一个诗词鉴定专家。"
USER_PROMPT_TEMPLATE = """以下共有6首诗词。请你作为一个整体进行评估，判断这一组诗词的来源。
这6首诗词是全部由人类创作的（判定为0），还是全部由人工智能生成的（判定为1）？

请先简单分析（100字以内），最后在末尾明确回答单字 0 或 1。

待评估列表：
{batch_content}"""

# ---------------- 🧠 核心逻辑 ----------------

def extract_answer(raw_text):
    if not raw_text or "Error" in str(raw_text): return raw_text
    clean = str(raw_text).strip().rstrip('。').lower()
    # 截取末尾文字进行判定
    target_zone = clean[-30:] 
    if any(word in target_zone for word in ['0', '人类']): return 0
    if any(word in target_zone for word in ['1', '人工智能']): return 1
    return 1 if '1' in clean else 0

def call_ai(prompt):
    for _ in range(3):
        try:
            res = client.chat.completions.create(
                model=cfg['model'],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"⚠️ 连接波动: {e}")
            time.sleep(5)
    return f"Error: {e}"

def process_batches(df_subset, label_prefix):
    results = []
    summary_list = []
    
    for i in range(0, len(df_subset), 6):
        batch = df_subset.iloc[i : i + 6].copy()
        
        # 补齐逻辑：不足6首时随机抽样补齐
        batch_to_send = batch
        if len(batch) < 6:
            padding = batch.sample(n=6 - len(batch), replace=True)
            batch_to_send = pd.concat([batch, padding])

        # 构建提示词
        batch_text = ""
        for idx, (_, row) in enumerate(batch_to_send.iterrows()):
            batch_text += f"No.{idx+1}: 《{row['title']}》\n{row['content']}\n\n"
        
        group_id = f"{label_prefix}_Group_{i//6 + 1}"
        print(f"🚀 [{CHOSEN_MODEL}] 正在处理: {group_id}")
        
        raw_res = call_ai(USER_PROMPT_TEMPLATE.format(batch_content=batch_text))
        final_ans = extract_answer(raw_res)
        
        summary_list.append({'组号': group_id, '结果': final_ans})
        
        for _, row in batch.iterrows():
            res_row = row.to_dict()
            res_row['group_id'] = group_id
            res_row['batch_result'] = final_ans
            results.append(res_row)
            
    return results, summary_list

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    df_all = pd.read_excel(INPUT_FILE)
    df_human = df_all[df_all['strategy'] == 'human'].copy()
    df_ai = df_all[df_all['strategy'] != 'human'].copy()
    
    print(f"开始执行 {CHOSEN_MODEL} 识别任务...")
    
    h_details, h_summary = process_batches(df_human, "Human")
    a_details, a_summary = process_batches(df_ai, "AI")
    
    # 保存结果
    pd.DataFrame(h_summary + a_summary).to_excel(SUMMARY_FILE, index=False)
    pd.DataFrame(h_details + a_details).to_excel(OUTPUT_FILE, index=False)
    
    print(f"\n🎉 任务完成！汇总：{SUMMARY_FILE}，详情：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()