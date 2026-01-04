import pandas as pd
from openai import OpenAI
import time, os
from tqdm import tqdm

# ---------------- 🚀 统一配置区域 ----------------
INPUT_FILE = 'Test.xlsx'
# 建议将 key 存放在环境变量中，此处留空
CONFIGS = {
    "DeepSeek": {
        "api_key": "YOUR_DEEPSEEK_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "wait_time": 0
    },
    "Kimi": {
        "api_key": "YOUR_KIMI_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2-turbo-preview",
        "wait_time": 3.2  # 应对 RPM 限制
    },
    "Doubao": {
        "api_key": "YOUR_DOUBAO_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "ep-xxxxxx", # 替换为你的 Endpoint ID
        "wait_time": 0
    },
    "GPT4": {
        "api_key": "YOUR_VECTOR_KEY",
        "base_url": "YOUR_BASE_URL",
        "model": "gpt-4.1",
        "wait_time": 0
    }
}

SYSTEM_PROMPT = "你是一个诗词鉴定专家。"
USER_PROMPT_TEMPLATE = "请判断这首诗是人类写的还是AI写的。请先简单分析，最后回答0（代表人类）或1（代表AI）。\n\n内容：\n《{title}》\n{content}"

# ---------------- 🧠 通用核心逻辑 ----------------

def extract_label(raw_text):
    """从回答末尾提取 0 或 1"""
    if not raw_text or "Error" in str(raw_text): return raw_text
    clean = str(raw_text).strip().rstrip('。').lower()
    zone = clean[-15:] # 重点检索区域
    
    if any(x in zone for x in ['0', '人类', 'human']): return 0
    if any(x in zone for x in ['1', '人工智能', 'ai']): return 1
    # 兜底全文本搜索
    if '0' in clean: return 0
    if '1' in clean: return 1
    return clean[:10] + "..."

def run_model_task(model_name):
    cfg = CONFIGS[model_name]
    output_file = f'Result_{model_name}.xlsx'
    client = OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'])

    # 1. 加载数据与断点续传
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
    else:
        df = pd.read_excel(INPUT_FILE)
        df['ans'] = None

    print(f"🚀 {model_name} 任务启动，总数: {len(df)}")

    # 2. 迭代处理
    for i, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
        if pd.notna(row['ans']) and "Error" not in str(row['ans']):
            continue
            
        prompt = USER_PROMPT_TEMPLATE.format(title=row['title'], content=row['content'])
        
        # 调用接口 (含重试)
        result = "Error: Failed"
        for attempt in range(3):
            try:
                if cfg['wait_time'] > 0: time.sleep(cfg['wait_time'])
                
                res = client.chat.completions.create(
                    model=cfg['model'],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                result = extract_label(res.choices[0].message.content)
                break
            except Exception as e:
                if "429" in str(e): time.sleep(10) # 限流冷却
                time.sleep(2 * (attempt + 1))
                result = f"Error: {e}"

        df.at[i, 'ans'] = result
        
        # 3. 定期保存
        if (i + 1) % 50 == 0:
            df.to_excel(output_file, index=False)

    df.to_excel(output_file, index=False)
    print(f"✅ {model_name} 处理完成！")

if __name__ == "__main__":

    target_model = "DeepSeek" 
    run_model_task(target_model)