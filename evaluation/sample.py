from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from common.annotation import tool
from inference.engine import Engine


@tool
def get_current_time(timezone: str):
    """获取当前日期和时间，支持指定时区
    Args:
        timezone: 时区名称，如Asia/Shanghai、America/New_York
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"无效的时区名称: {timezone}", "timezone": timezone}
    now = datetime.now(tz)
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}


@tool
def get_weather(city: str, date: str = "today"):
    """查询指定城市某天的天气
    Args:
        city: 城市名称，如北京、上海
        date: 日期，格式YYYY-MM-DD，默认today表示今天
    """
    return {"city": city, "date": date, "weather": "晴", "temperature": "22°C"}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str):
    """货币汇率换算
    Args:
        amount: 金额
        from_currency: 源货币代码，如CNY、USD
        to_currency: 目标货币代码，如CNY、USD
    """
    return {"amount": amount, "from": from_currency, "to": to_currency, "result": amount * 7.1}


@tool
def calculate(expression: str):
    """计算一个数学表达式并返回结果
    Args:
        expression: 数学表达式，如 (3 + 5) * 2
    """
    try:
        return {"expression": expression, "result": eval(expression, {"__builtins__": {}})}
    except Exception as e:
        return {"error": str(e), "expression": expression}


@tool
def send_email(to: str, subject: str, body: str):
    """发送一封电子邮件
    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
    """
    return {"status": "sent", "to": to, "subject": subject}


@tool
def search_flight(origin: str, destination: str, date: str):
    """查询航班信息
    Args:
        origin: 出发城市
        destination: 到达城市
        date: 出发日期，格式YYYY-MM-DD
    """
    return {"origin": origin, "destination": destination, "date": date, "flights": ["CA1234 08:00", "MU5678 14:30"]}


# 每条: (system_prompt, user_query, [可用工具])
test_case = [

    # --- 1. 单工具直接调用 ---
    ("", "现在上海几点了", [get_current_time]),

    # --- 2. 城市→时区的隐式推理（模型需自己映射到America/New_York）---
    ("", "纽约现在是几点？", [get_current_time]),

    # --- 3. 缺省参数处理（date有默认值，不该硬填）---
    ("", "北京今天天气怎么样", [get_weather]),

    # --- 4. 数值类型参数提取 ---
    ("", "帮我把100美元换算成人民币", [convert_currency]),

    # --- 5. 多工具并存时的正确选择（有多个工具，只应选calculate）---
    ("", "帮我算一下 (18 + 6) * 3 等于多少", [get_current_time, get_weather, calculate]),

    # --- 6. 多参数一次性抽取 ---
    ("", "给 zhangsan@example.com 发封邮件，主题是周报，内容写“本周任务已完成”", [send_email]),

    # --- 7. 需要多次工具调用（上海和东京两个时区）---
    ("", "上海和东京现在分别几点？", [get_current_time]),

    # --- 8. 跨工具组合任务（查完天气/时间后要不要继续）---
    ("", "查一下明天从北京到上海的航班", [search_flight, get_weather]),

    # --- 9. 不需要调用工具（闲聊，考察模型不过度调用）---
    ("", "你好，你能做什么？", [get_current_time, get_weather, calculate]),

    # --- 10. 工具无法满足需求（应说明无法完成而非强行调用）---
    ("", "帮我预订一张明晚的电影票", [get_current_time, get_weather, calculate]),

    # --- 11. 参数缺失，应追问而非编造（没给收件人）---
    ("", "帮我发封邮件说我今天请假", [send_email]),

    # --- 12. 带 system prompt 的约束场景 ---
    ("你是一个只用UTC时区回答的助手。", "现在几点了", [get_current_time]),

    # --- 13. 多轮/链式：先算汇率再判断（考察是否能连续调用）---
    ("", "300欧元换成美元大概是多少？再帮我算算这些钱能买几张120美元的门票", [convert_currency, calculate]),

    ("","你是什么模型？",[])
]

def sample_chat(model,tokeniz,max_len,device_type):
    engine=Engine(model,tokeniz,device_type)
    for i,case in enumerate(test_case):
        print(f"case [{i}/{len(test_case)}]")
        sentence=engine.generate_chat(case[0],case[1],case[2],max_len)
        print(sentence)
