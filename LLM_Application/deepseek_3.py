import os
from os import getenv
import re
from openai import OpenAI

def clean_and_tokenize(text):
    # 去除网址
    text = re.sub(r'https?://\S+', '', text)
    # 去除时间戳和用户名
    text = re.sub(r'\w+_\w+ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', text)
    # 去除多余标点和特殊符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text)
    return text

def deepseek(chat_text):
    """
    Args:
        chat_text (char): 输入的客服用户聊天文本

    Returns:
        char : 返回的总结
    """
    deepseek_api = os.getenv("DEEPSEEK_API")
    client = OpenAI(api_key = deepseek_api, base_url="https://api.deepseek.com/v1")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system", 
                "content" : '''
                            你是一个专业的电商客服对话分析专家。你的任务是根据用户与客服的聊天记录，忽略用户选择的“买多/买错/不想要”，“尺码没选对”等表面退款理由，精准判断并总结出用户退款的真实原因。
                            **请严格遵循以下步骤和输出规则：**
                            1.  **判断对话阶段**：
                                首先，分析聊天记录，判断对话发生在用户**购买前**还是**购买后**。
                                *   **购买前**的对话特征：咨询商品信息、库存、优惠、发货时效等，但未提及已下单或已收到货。
                                *   **购买后**的对话特征：提供订单号，或提及“已收到货”、“试了一下”、“申请退款”、“怎么退货”等关键词。
                            2.  **根据阶段输出结果**：
                                *   **如果判断为【购买前】**：
                                    你的唯一输出应该是固定文本：“购买前咨询问题”。
                                *   **如果判断为【购买后】**：
                                    你需要从下方的【原因分类列表】中，选择一个最符合聊天内容的分类、并将具体问题总结为**5个字以内**的原因进行输出。
                            3.  **【原因分类列表】**
                                *   尺码偏大
                                *   尺码偏小
                                *   发货/配送慢
                                *   价格问题
                                *   颜色/款式不合适
                                *   商品破损
                                *   包装问题
                                *   少件
                                *   发错货
                                *   未收到货
                                *   发票问题
                                *   商品与页面描述不符
                                *   7天无理由 (当用户在对话中也未给出明确具体原因时使用)
                            4.  **【特殊规则：商品质量问题】**
                                *   如果聊天内容明确指向商品存在**质量问题**（例如：开线、掉色、面料起球、拉链损坏、有异味、功能失效等），**请不要**简单地输出“质量问题”。
                                *   你需要**直接总结出具体的质量问题**，例如：“**面料起球**”、“**拉链损坏**”、“**衣服开线**”、“**鞋子开胶**”。这种具体描述**可以超过5个字**，但需力求简洁。
                            '''
            },
            
            {"role": "user", "content": chat_text},
        ],
        stream=False
    )
    result = response.choices[0].message.content
    return result