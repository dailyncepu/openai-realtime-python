import asyncio
import base64
import json
import logging
import os
import aiohttp

from typing import Any, AsyncGenerator
from .struct import InputAudioBufferAppend, ClientToServerMessage, ServerToClientMessage, parse_server_message, to_json
from ..logger import setup_logger

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)


DEFAULT_VIRTUAL_MODEL = "gpt-4o-realtime-preview"

def smart_str(s: str, max_field_len: int = 128) -> str:
    """parse string as json, truncate data field to 128 characters, reserialize"""
    try:
        data = json.loads(s)
        if "delta" in data:
            key = "delta"
        elif "audio" in data:
            key = "audio"
        else:
            return s

        if len(data[key]) > max_field_len:
            data[key] = data[key][:max_field_len] + "..."
        return json.dumps(data)
    except json.JSONDecodeError:
        return s


class RealtimeApiConnection:
    """
        该connection.py文件管理代理与 OpenAI API 之间的实时通信。它处理连接设置、发送和接收消息以及管理音频数据流。
        该类RealtimeApiConnection封装了所有连接逻辑，使集成实时 AI 响应变得更加容易。
        此连接生命周期管理对于处理实时应用程序中的长时间运行的 WebSocket 会话至关重要。
    """
    def __init__(
        self,
        base_uri: str,
        api_key: str | None = None,
        path: str = "/v1/realtime",
        verbose: bool = False,
        model: str = DEFAULT_VIRTUAL_MODEL,
    ):
        """
        在初始化期间，OpenAI 密钥、API URL（包括模型）和身份验证令牌将传递给客户端，并初始化 WebSocket 会话。
        """
        
        self.url = f"{base_uri}{path}"
        if "model=" not in self.url:
            self.url += f"?model={model}"

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.verbose = verbose
        self.session = aiohttp.ClientSession()

    async def __aenter__(self) -> "RealtimeApiConnection":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        await self.close()
        return False

    async def connect(self):
        """
        该connect方法使用身份验证标头建立与指定 URL 的 WebSocket 连接。该close方法确保 WebSocket 连接正常关闭，从而防止资源泄漏。此连接生命周期管理对于处理实时应用程序中的长时间运行的 WebSocket 会话至关重要。
        """
        auth = aiohttp.BasicAuth("", self.api_key) if self.api_key else None

        headers = {"OpenAI-Beta": "realtime=v1"}

        self.websocket = await self.session.ws_connect(
            url=self.url,
            auth=auth,
            headers=headers,
        )

    async def send_audio_data(self, audio_data: bytes):
        """
            audio_data is assumed to be pcm16 24kHz mono little-endian
            通过 WebSocket 发送音频数据（以 base64 编码）。它将音频数据打包成一个ClientToServerMessage并调用send_request以传输它
        """
        base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
        message = InputAudioBufferAppend(audio=base64_audio_data)
        await self.send_request(message)

    async def send_request(self, message: ClientToServerMessage):
        """方法记录传出消息（如果启用了详细日志记录）并通过 WebSocket 连接发送它"""
        assert self.websocket is not None
        message_str = to_json(message)
        if self.verbose:
            logger.info(f"-> {smart_str(message_str)}")
        await self.websocket.send_str(message_str)

    

    async def listen(self) -> AsyncGenerator[ServerToClientMessage, None]:
        """
        该listen方法监听来自 WebSocket 的传入消息。它使用异步生成器以非阻塞方式处理传入消息。
        根据消息类型（文本或错误），它会处理消息并将其传递给handle_server_message。
        如果启用了详细日志记录，则会记录传入消息以方便调试。
        """
        assert self.websocket is not None
        if self.verbose:
            logger.info("Listening for realtimeapi messages")
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if self.verbose:
                        logger.info(f"<- {smart_str(msg.data)}")
                    yield self.handle_server_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Error during receive: %s", self.websocket.exception())
                    break
        except asyncio.CancelledError:
            logger.info("Receive messages task cancelled")

    def handle_server_message(self, message: str) -> ServerToClientMessage:
        """
        该handle_server_message方法解析服务器的消息并处理解析过程中发生的任何异常。
        此方法可确保将格式错误的消息记录为错误，从而帮助追踪服务器响应格式的问题。
        """
        try:
            return parse_server_message(message)
        except Exception as e:
            logger.error("Error handling message: " + str(e))
            raise e

    async def close(self):
        """
        该close方法确保 WebSocket 连接正常关闭，从而防止资源泄漏。
        """
        # Close the websocket connection if it exists
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
