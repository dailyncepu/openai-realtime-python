
from typing import Any
from realtime_agent.tools import ToolContext
from realtime_agent.realtime.function import weather,emotion
# Function calling Example
# This is an example of how to add a new function to the agent tools.

class AgentTools(ToolContext):
    def __init__(self) -> None:
        super().__init__()
        
        # create multiple functions here as per requirement
        self.register_function(
            name = "getImageInfo",
            # description="当用户的提问与图片信息相关时(如：发型，衣服，着装 等信息)，调用该函数获取图片信息(不需要用户输入图片)，函数输入只有图片的处理指令。",
            description="当用户提问与图片信息相关时，调用该函数获取图片信息。如：询问当前样貌，当前场景等。",
            parameters= {
                "type": "object",
                "properties": {
                    "img_prompt": {
                        "type": "string",
                        "description": "对图片处理的指令",
                    },
                },
                "required": ["img_prompt"],
            },
            ##函数
            fn = self.getImageInfo
        )

        self.register_function(
            name = "getweather",
            description="查询某个城市的天气信息",
            parameters= {
                "type": "object",
                "properties": {
                    "city_code": {
                    "type": "string",
                    "description": "城市代码（例如北京是 101010100）",
                    },
                    "n":{
                    "type" : "integer",
                    "description": "用来推算是哪一天。取值范围[-1,7]，-1代表昨天，0代表今天，1代表明天，2代表后天 等。",
                    }
                },
                "required": ["city_code","n"],
            },
            ##函数
            fn=self.getweather,
        )
        
        self.register_function(
            name = "getEmotion",
            #description="每次输入都需要调用该函数，根据用户输入的内容信息，判断回复用户时，使用的情绪标签。",
            description="每次输入都需要调用该函数，判断输入音频的情绪标签，同时判断应该以哪种情绪标签回复。",
            parameters= {
                "type": "object",
                "properties": {
                    "user_emotion": {
                        "type": "string",
                        "description": "输入音频的情绪标签。",
                        "enum": ["快乐","悲伤","好奇","惊讶","困倦","生气","害怕","困惑","自豪","害羞","期待","信任","厌恶","安抚","平和"]
                    },
                    "response_emotion":{
                        "type": "string",
                        "description": "模型即将回复时，使用的情绪标签。例如：1.输入是悲伤情绪时，反馈的标签可以是 悲伤，也可以是安抚。2. 输入是快乐时，可以是快乐和好奇 等。",
                        "enum": ["快乐","悲伤","好奇","惊讶","困倦","生气","害怕","困惑","自豪","害羞","期待","信任","厌恶","安抚","平和"]
                    }
                },
                "required": ["user_emotion","response_emotion"],
            },
            ##函数
            fn = self.getEmotion
        )
    async def getweather(self,city_code, n):
        
        return weather.get_weather_by_city_code(city_code,n)

    async def getEmotion(sefl,user_emotion,response_emotion):
        
        return emotion.get_respond_emotion(user_emotion,response_emotion)

    async def getImageInfo(self,img_prompt):
        
        return "不清楚图片上的信息是啥。"