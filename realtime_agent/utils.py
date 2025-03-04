import asyncio
import functools
from datetime import datetime
import requests
import json
import base64
def write_pcm_to_file(buffer: bytearray, file_name: str) -> None:
    """Helper function to write PCM data to a file."""
    with open(file_name, "ab") as f:  # append to file
        f.write(buffer)


def generate_file_name(prefix: str) -> str:
    # Create a timestamp for the file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.pcm"


class PCMWriter:
    def __init__(self, prefix: str, write_pcm: bool, buffer_size: int = 1024 * 64):
        self.write_pcm = write_pcm
        self.buffer = bytearray()
        self.buffer_size = buffer_size
        self.file_name = generate_file_name(prefix) if write_pcm else None
        self.loop = asyncio.get_event_loop()

    async def write(self, data: bytes) -> None:
        """Accumulate data into the buffer and write to file when necessary."""
        if not self.write_pcm:
            return

        self.buffer.extend(data)

        # Write to file if buffer is full
        if len(self.buffer) >= self.buffer_size:
            await self._flush()

    async def flush(self) -> None:
        """Write any remaining data in the buffer to the file."""
        if self.write_pcm and self.buffer:
            await self._flush()

    async def _flush(self) -> None:
        """Helper method to write the buffer to the file."""
        if self.file_name:
            await self.loop.run_in_executor(
                None,
                functools.partial(write_pcm_to_file, self.buffer[:], self.file_name),
            )
        self.buffer.clear()


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
def call_vllm_via_base64(url,img_base64,prompt,max_token = 30, temperature = 0.3, top_p = 0.7):

    api_key = ""
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {api_key}"}


    contentList = [{"type": "text", "text": prompt}]
    ndict = {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    contentList.append(ndict)

    payload ={
        "model": "riskchat",
        "messages": [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content":contentList},
                ],
        "max_tokens": int(max_token),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }

    try :
        return_info = requests.post(url, headers=headers, json=payload)
        if  return_info.status_code != 200:
            return 100,"不知道图片中描述的信息。"
        res = json.loads(return_info.text)
        return 200,res["choices"][0]["message"]["content"]
    except:
        return 100,"不知道图片中描述的信息。"

# if __name__=="__main__":
#     imgpath = "/home/work/slg/realtime/img/1.jpg"
#     imgprompt = "你看看我的发型咋样啊？"
#     imgbase64 = image_to_base64(imgpath)
#     res = call_vllm_via_base64("http://39.97.186.121:58084/v1/chat/completions",imgbase64,imgprompt,max_token = 30, temperature = 0.3, top_p = 0.7)
#     print(res)

