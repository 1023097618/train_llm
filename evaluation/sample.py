from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from common.annotation import tool
from inference.engine import Engine


@tool
def get_current_time(timezone: str):
    """Get the current date and time in a given timezone
    Args:
        timezone: IANA timezone name, e.g. Asia/Shanghai, America/New_York
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"invalid timezone: {timezone}", "timezone": timezone}
    now = datetime.now(tz)
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}


@tool
def get_weather(city: str, date: str = "today"):
    """Get the weather of a city on a given day
    Args:
        city: city name, e.g. Beijing, London
        date: date in YYYY-MM-DD format, defaults to today
    """
    return {"city": city, "date": date, "weather": "sunny", "temperature": "22°C"}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str):
    """Convert an amount of money from one currency to another
    Args:
        amount: amount of money
        from_currency: source currency code, e.g. CNY, USD
        to_currency: target currency code, e.g. CNY, USD
    """
    return {"amount": amount, "from": from_currency, "to": to_currency, "result": amount * 7.1}


@tool
def calculate(expression: str):
    """Evaluate a math expression and return the result
    Args:
        expression: math expression, e.g. (3 + 5) * 2
    """
    try:
        return {"expression": expression, "result": eval(expression, {"__builtins__": {}})}
    except Exception as e:
        return {"error": str(e), "expression": expression}


@tool
def send_email(to: str, subject: str, body: str):
    """Send an email
    Args:
        to: recipient email address
        subject: email subject
        body: email body
    """
    return {"status": "sent", "to": to, "subject": subject}


@tool
def search_flight(origin: str, destination: str, date: str):
    """Search for flights
    Args:
        origin: departure city
        destination: arrival city
        date: departure date in YYYY-MM-DD format
    """
    return {"origin": origin, "destination": destination, "date": date, "flights": ["CA1234 08:00", "MU5678 14:30"]}


# 每条: (system_prompt, user_query, [可用工具])
test_case = [

    # --- 1. 单工具直接调用 ---
    ("", "What time is it in Shanghai right now?", [get_current_time]),

    # --- 2. 城市→时区的隐式推理（模型需自己映射到America/New_York）---
    ("", "What time is it in New York?", [get_current_time]),

    # --- 3. 缺省参数处理（date有默认值，不该硬填）---
    ("", "What's the weather like in Beijing today?", [get_weather]),

    # --- 4. 数值类型参数提取 ---
    ("", "Convert 100 US dollars to Chinese yuan for me.", [convert_currency]),

    # --- 5. 多工具并存时的正确选择（有多个工具，只应选calculate）---
    ("", "What is (18 + 6) * 3?", [get_current_time, get_weather, calculate]),

    # --- 6. 多参数一次性抽取 ---
    ("", "Send an email to zhangsan@example.com with the subject \"Weekly report\" and the body \"All tasks for this week are done.\"", [send_email]),

    # --- 7. 需要多次工具调用（上海和东京两个时区）---
    ("", "What time is it in Shanghai and in Tokyo right now?", [get_current_time]),

    # --- 8. 跨工具组合任务（查完天气/时间后要不要继续）---
    ("", "Find me flights from Beijing to Shanghai tomorrow.", [search_flight, get_weather]),

    # --- 9. 不需要调用工具（闲聊，考察模型不过度调用）---
    ("", "Hi, what can you do?", [get_current_time, get_weather, calculate]),

    # --- 10. 工具无法满足需求（应说明无法完成而非强行调用）---
    ("", "Book me a movie ticket for tomorrow night.", [get_current_time, get_weather, calculate]),

    # --- 11. 参数缺失，应追问而非编造（没给收件人）---
    ("", "Send an email saying I'm taking the day off today.", [send_email]),

    # --- 12. 带 system prompt 的约束场景 ---
    ("You are an assistant that always answers in the UTC timezone.", "What time is it now?", [get_current_time]),

    # --- 13. 多轮/链式：先算汇率再判断（考察是否能连续调用）---
    ("", "Roughly how much is 300 euros in US dollars? Then work out how many 120-dollar tickets that money can buy.", [convert_currency, calculate]),

    ("", "What model are you?", [])
]

def sample_chat(model,tokeniz,max_len,device_type):
    engine=Engine(model,tokeniz,device_type)
    for i,case in enumerate(test_case):
        print(f"case [{i}/{len(test_case)}]")
        sentence=engine.generate_chat(case[0],case[1],case[2],max_len)
        print(sentence)
