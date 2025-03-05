import asyncio
import functools
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
import json
import requests
import base64
import bisect
import io
import time
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from functools import lru_cache
from agora.rtc.video_frame_observer import VideoFrame


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


@dataclass(kw_only=True)
class VideoFrameData:
    type: str = "rgb"
    width: int = 0
    height: int = 0
    data: bytearray = None
    base64_str: str = None
    timestamp: int = 0


class VFrameFormatConverter:
    def __init__(self):
        # 预计算 YUV 到 RGB 的转换矩阵
        self.yuv2rgb_matrix = np.array([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ])

    @lru_cache(maxsize=8)
    def _create_conversion_matrices(self, height: int, width: int) -> tuple[NDArray, NDArray, NDArray]:
        """缓存转换矩阵以提高性能"""
        y_matrix = np.zeros((height, width), dtype=np.float32)
        u_matrix = np.zeros((height, width), dtype=np.float32)
        v_matrix = np.zeros((height, width), dtype=np.float32)
        return y_matrix, u_matrix, v_matrix

    async def yuv420_to_rgb_v2(self, frame: VideoFrame, save_path: str = None) -> VideoFrameData: # TODO: 颜色模式问题定位
        height, width = frame.height, frame.width
        
        # 使用预分配的矩阵
        y_matrix, u_matrix, v_matrix = self._create_conversion_matrices(height, width)
        
        # 优化 YUV 数据处理
        y_matrix = np.frombuffer(frame.y_buffer, dtype=np.uint8).reshape(height, frame.y_stride)[:, :width]
        u_temp = np.frombuffer(frame.u_buffer, dtype=np.uint8).reshape(height//2, frame.u_stride)[:, :width//2]
        v_temp = np.frombuffer(frame.v_buffer, dtype=np.uint8).reshape(height//2, frame.v_stride)[:, :width//2]

        # 使用向量化操作进行上采样
        u_matrix = np.kron(u_temp - 128, np.ones((2, 2), dtype=np.float32))
        v_matrix = np.kron(v_temp - 128, np.ones((2, 2), dtype=np.float32))
        
        # 向量化 YUV 到 RGB 的转换
        yuv = np.stack([y_matrix, u_matrix, v_matrix], axis=-1)
        rgb = np.clip(np.dot(yuv, self.yuv2rgb_matrix.T), 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, 'RGB')

        vframe = VideoFrameData(
            type="rgb",
            width=width,
            height=height,
            base64_str=image_pil_to_base64(image),
            timestamp=frame.render_time_ms
        )

        if save_path:
            try:
                image.save(save_path, quality=95, optimize=True)
            except Exception as e:
                print(f"Error saving frame to {save_path}: {e}")
        return vframe

    async def yuv420_to_rgb(self, frame: VideoFrame, save_path: str = None) -> VideoFrameData: # TODO: 性能测试
        """
        将YUV格式的VideoFrame转换为RGB格式，使用PIL处理

        参数:
            frame: VideoFrame对象，包含YUV数据和相关参数
            save_path: 保存图片的路径，如果为None则不保存

        返回:
            RGB格式的字节数据
        """
        height, width = frame.height, frame.width

        # 转换YUV数据
        Y = np.frombuffer(frame.y_buffer, dtype=np.uint8).reshape(height, frame.y_stride)[:, :width]
        U = np.frombuffer(frame.u_buffer, dtype=np.uint8).reshape(height//2, frame.u_stride)[:, :width//2]
        V = np.frombuffer(frame.v_buffer, dtype=np.uint8).reshape(height//2, frame.v_stride)[:, :width//2]

        # 上采样U和V分量
        U = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)
        V = np.repeat(np.repeat(V, 2, axis=1), 2, axis=0)

        # YUV到RGB的转换
        Y = Y.astype(np.float32)
        U = U.astype(np.float32) - 128
        V = V.astype(np.float32) - 128

        R = Y + 1.402 * V
        G = Y - 0.344136 * U - 0.714136 * V
        B = Y + 1.772 * U

        # 裁剪值到0-255范围
        RGB = np.clip(np.dstack([R, G, B]), 0, 255).astype(np.uint8)
        image = Image.fromarray(RGB, 'RGB')

        vframe = VideoFrameData(
            type="rgb",
            width=width,
            height=height,
            base64_str=image_pil_to_base64(image),
            timestamp=frame.render_time_ms
        )

        if save_path:
            try:
                image.save(save_path, quality=95, optimize=True)
            except Exception as e:
                print(f"Error saving frame to {save_path}: {e}")
        return vframe


class VFrameSynchronizer:
    def __init__(self, buffer_size=30, sync_threshold=100, cleanup_interval=1000):
        self.buffer_size = buffer_size
        self.sync_threshold = sync_threshold
        self.cleanup_interval = cleanup_interval
        self.frame_index = OrderedDict()
        self.last_cleanup_time = 0
        self.lock = asyncio.Lock()
        self._sorted_timestamps = []
        self._needs_sort = False

    async def _cleanup_expired_frames(self, current_time: int) -> None:
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return

        expiration_time = current_time - (2 * self.sync_threshold)
        
        # 使用二分查找找到过期帧的位置
        if self._needs_sort:
            self._sorted_timestamps = sorted(self.frame_index.keys())
            self._needs_sort = False
            
        idx = bisect.bisect_right(self._sorted_timestamps, expiration_time)
        if idx > 0:
            expired_ts = self._sorted_timestamps[:idx]
            for ts in expired_ts:
                del self.frame_index[ts]
            self._sorted_timestamps = self._sorted_timestamps[idx:]
        
        self.last_cleanup_time = current_time

    async def add_video_frame(self, video_frame: VideoFrameData):
        async with self.lock:
            current_time = video_frame.timestamp
            await self._cleanup_expired_frames(current_time)
            
            self.frame_index[current_time] = video_frame
            self._needs_sort = True
            
            if len(self.frame_index) > self.buffer_size:
                self.frame_index.popitem(last=False)
                self._needs_sort = True

    async def find_matching_frame(self, target_time: int) -> VideoFrameData | None:
        async with self.lock:
            if not self.frame_index:
                return None

            if self._needs_sort:
                self._sorted_timestamps = sorted(self.frame_index.keys())
                self._needs_sort = False

            window_start = target_time - self.sync_threshold
            window_end = target_time + self.sync_threshold

            # 使用二分查找确定时间窗口范围
            left_idx = bisect.bisect_left(self._sorted_timestamps, window_start)
            right_idx = bisect.bisect_right(self._sorted_timestamps, window_end)

            if left_idx >= len(self._sorted_timestamps):
                return None

            # 在窗口范围内找到最接近的帧
            closest_ts = min(
                self._sorted_timestamps[left_idx:right_idx],
                key=lambda x: abs(x - target_time),
                default=None
            )

            return self.frame_index.get(closest_ts)


def image_bytes_to_base64(image_bytes, file_prefix=None):
    # 将图像数据转换为 PIL.Image 对象
    image = Image.fromarray(np.frombuffer(image_bytes, dtype=np.uint8).reshape(640, 480, 3), 'RGB')
    # 创建一个 BytesIO 对象来存储图像数据
    buffered = io.BytesIO()
    # 将图像保存到 BytesIO 对象中
    image.save(buffered, format="JPEG")  # 你可以根据需要选择其他格式，如 "PNG"
    # 获取图像的二进制数据
    img_bytes = buffered.getvalue()
    # 对二进制数据进行 Base64 编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    if file_prefix:
        image.save(f"frame_{file_prefix}_{time.time()*1000}.png", quality=95, optimize=True)
    return img_base64

def image_pil_to_base64(image, file_prefix=None):
    """
    将PIL的Image对象转换为Base64编码的字符串。
    
    参数:
        image (PIL.Image): PIL的Image对象。
    
    返回:
        str: Base64编码的字符串。
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # 保存为PNG格式
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    if file_prefix:
        image.save(f"frame_{file_prefix}_{time.time()*1000}.png", quality=95)
    return img_base64

def base64_to_pil_image(base64_str, file_prefix=None):
    """
    将Base64编码的字符串转换为PIL的Image对象。
    
    参数:
        base64_str (str): Base64编码的字符串。
    
    返回:
        PIL.Image: PIL的Image对象。
    """
    img_bytes = base64.b64decode(base64_str)
    buffered = io.BytesIO(img_bytes)
    image = Image.open(buffered)
    if file_prefix:
        image.save(f"frame_{file_prefix}_{time.time()*1000}.png", quality=95, optimize=True)
    return image


def image_file_to_base64(image_input):
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


async def call_vllm_via_base64(url,img_base64,prompt,max_token = 30, temperature = 0.3, top_p = 0.7):
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
        desc_info, ret_msg = "不知道图片中描述的信息。", "success"
        return_info = requests.post(url, headers=headers, json=payload)
        if  return_info.status_code != 200:
            ret_msg = f"failed, status code is not 200, detail:{return_info}"
            return 100, desc_info, ret_msg
        res = json.loads(return_info.text)
        desc_info = res["choices"][0]["message"]["content"]
        return 200, desc_info, ret_msg
    except Exception as e:
        ret_msg = str(e)
        return 100, desc_info, ret_msg

# if __name__=="__main__":
#     imgpath = "/home/work/slg/realtime/img/1.jpg"
#     imgprompt = "你看看我的发型咋样啊？"
#     imgbase64 = image_to_base64(imgpath)
#     res = call_vllm_via_base64("http://39.97.186.121:58084/v1/chat/completions",imgbase64,imgprompt,max_token = 30, temperature = 0.3, top_p = 0.7)
#     print(res)

