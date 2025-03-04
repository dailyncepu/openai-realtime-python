from openai import OpenAI
import base64
import requests
import json
import time
import traceback
import os
from dotenv import load_dotenv


#阿里
#client = OpenAI(api_key = "",base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

#gpu05  deepseek-tiny
load_dotenv()
api_key = os.getenv("TMP_API_KEY")
print(api_key)
client = OpenAI(api_key = api_key,base_url = "http://39.97.186.121:58084/v1")

def image_to_base64(image_input):
    # 判断输入是 URL 还是文件路径
    if image_input.startswith("http") or image_input.startswith("https"):
        # 从 URL 读取图片
        response = requests.get(image_input)
        if response.status_code == 200:
            image_data = response.content
        else:
            raise ValueError(f"Failed to fetch image from URL: {image_input}")
    else:
        # 从文件路径读取图片
        with open(image_input, "rb") as image_file:
            image_data = image_file.read()

    # 将图片数据转换为 Base64 编码
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded
def getImageInfo(image_base64_list, img_prompt):

    contentList = [{"type": "text", "text": img_prompt}]
    try:
        for img_base64 in image_base64_list:
            ndict = {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            contentList.append(ndict)
        response = client.chat.completions.create(
            model="riskchat",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content":contentList}
            ],
            max_tokens=30,
            temperature=0.2,
            top_p=1,
            stream=False,
        )
        return response.choices[0].message.content
    except:
        traceback.print_exc()
        return "不知道图片中描述的信息。"

if __name__=="__main__":
    imgpath = "/home/work/slg/realtime/img/1.jpg"
    imgprompt = "你看看我的发型咋样啊？"
    imgbase64 = image_to_base64(imgpath)
    res = getImageInfo([imgbase64],imgprompt)
    print(res)



