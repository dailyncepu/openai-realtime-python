import requests
from datetime import datetime, timedelta

def get_previous_date(n = 0):
    today = datetime.now()
    previous_date = today + timedelta(days=n)
    previous_date_str = previous_date.strftime("%Y-%m-%d")
    return previous_date_str

def parser_date(datajson):
    datadict = {}
    city = datajson["cityInfo"]["city"]
    forecast = datajson["data"]["forecast"]
    forecast.append(datajson["data"]["yesterday"])

    for e in forecast:
        tq = e["type"]
        high = e["high"]
        low = e["low"]
        fx = e["fx"]
        fl = e["fl"]
        date = e["ymd"]
        elem = "日期：" + date +"  天气："+tq + "  温度：" + high+" " + low +  "  风型："+fx +"  风力：" + fl 
        datadict[date] = elem  
    return datadict,city

def get_weather_by_city_code(city_code, n):
    """
    根据城市代码查询天气预报

    :param city_code: 城市代码（例如 "101010100" 表示北京）
    :return: 返回天气预报信息的字典，如果查询失败则返回 None
    """
    base_url = f"http://t.weather.itboy.net/api/weather/city/{city_code}"
    date = get_previous_date(n)
    try:
        # 发送 GET 请求
        response = requests.get(base_url)
        response.raise_for_status()  # 检查请求是否成功
        # 解析返回的 JSON 数据
        weather_data = response.json()
        # 提取需要的天气信息
        if weather_data.get("status") == 200:  # 判断请求是否成功
            datadict,city = parser_date(weather_data)
            return city + "天气预报结果：\n" + datadict[date]
        else:
            print("请求失败，返回状态码:", weather_data.get("status"))
            return "天气查询失败"
    
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return "天气查询失败"

# 示例用法
if __name__ == "__main__":
    # 示例城市代码：101010100 表示北京
    city_code = "101010100"
    
    # 查询天气预报
    weather = get_weather_by_city_code(city_code,0)
    
    print(weather)