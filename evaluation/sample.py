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
    # same safety rules as nanochat's calculator: plain arithmetic only, no power operator, time limit,
    # so a sampled "9**9**9" can neither hang the process nor return a number with thousands of digits
    from inference.engine import eval_with_timeout
    expr = str(expression).strip()
    if not expr or any(ch not in "0123456789+-*/.() " for ch in expr):
        return {"error": "only numbers and + - * / ( ) are allowed", "expression": expression}
    if "**" in expr:
        return {"error": "the power operator is not allowed", "expression": expression}
    result = eval_with_timeout(expr, max_time=3)
    if result is None:
        return {"error": "could not evaluate the expression", "expression": expression}
    if len(str(result)) > 40:
        return {"error": "the result is too large", "expression": expression}
    return {"expression": expression, "result": result}


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


# each case: (system_prompt, user_query, [available tools])
test_case = [

    # --- 1. one tool, called directly ---
    ("", "What time is it in Shanghai right now?", [get_current_time]),

    # --- 2. city -> time zone, implicit (the model has to map it to America/New_York itself) ---
    ("", "What time is it in New York?", [get_current_time]),

    # --- 3. default arguments (date has a default and should not be filled in) ---
    ("", "What's the weather like in Beijing today?", [get_weather]),

    # --- 4. extracting numeric arguments ---
    ("", "Convert 100 US dollars to Chinese yuan for me.", [convert_currency]),

    # --- 5. choosing among several tools (only calculate fits) ---
    ("", "What is (18 + 6) * 3?", [get_current_time, get_weather, calculate]),

    # --- 6. several arguments at once ---
    ("", "Send an email to zhangsan@example.com with the subject \"Weekly report\" and the body \"All tasks for this week are done.\"", [send_email]),

    # --- 7. several calls needed (Shanghai and Tokyo time zones) ---
    ("", "What time is it in Shanghai and in Tokyo right now?", [get_current_time]),

    # --- 8. combining tools (whether to go on after the weather / time lookup) ---
    ("", "Find me flights from Beijing to Shanghai tomorrow.", [search_flight, get_weather]),

    # --- 9. no tool needed (small talk: the model should not call tools needlessly) ---
    ("", "Hi, what can you do?", [get_current_time, get_weather, calculate]),

    # --- 10. no tool fits (it should say so rather than force a call) ---
    ("", "Book me a movie ticket for tomorrow night.", [get_current_time, get_weather, calculate]),

    # --- 11. a missing argument: ask instead of inventing it (no recipient given) ---
    ("", "Send an email saying I'm taking the day off today.", [send_email]),

    # --- 12. constraints in a system prompt ---
    ("You are an assistant that always answers in the UTC timezone.", "What time is it now?", [get_current_time]),

    # --- 13. chained: convert the currency first, then decide (several calls in a row) ---
    ("", "Roughly how much is 300 euros in US dollars? Then work out how many 120-dollar tickets that money can buy.", [convert_currency, calculate]),

    ("", "What model are you?", [])
]

def sample_chat(model,tokeniz,max_len,device_type):
    engine=Engine(model,tokeniz,device_type)
    for i,case in enumerate(test_case):
        print(f"case [{i}/{len(test_case)}]")
        sentence=engine.generate_chat(case[0],case[1],case[2],max_len)
        print(sentence)
