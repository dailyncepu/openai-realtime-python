import asyncio
import base64
import logging
import os
from builtins import anext
from typing import Any
import json
from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from attr import dataclass

from .mp_rtc import Channel, ChatMessage, RtcEngine, RtcOptions

from .logger import setup_logger
from .realtime.struct import ErrorMessage, FunctionCallOutputItemParam, InputAudioBufferCommitted, InputAudioBufferSpeechStarted, InputAudioBufferSpeechStopped, InputAudioTranscription, ItemCreate, ItemCreated, ItemInputAudioTranscriptionCompleted, RateLimitsUpdated, ResponseAudioDelta, ResponseAudioDone, ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone, ResponseContentPartAdded, ResponseContentPartDone, ResponseCreate, ResponseCreated, ResponseDone, ResponseFunctionCallArgumentsDelta, ResponseFunctionCallArgumentsDone, ResponseOutputItemAdded, ResponseOutputItemDone, ServerVADUpdateParams, SessionUpdate, SessionUpdateParams, SessionUpdated, Voices, to_json
from .realtime.connection import RealtimeApiConnection
from .tools import ClientToolCallResponse, ToolContext
from .utils import PCMWriter
from agora.rtc.video_frame_observer import VideoFrame 

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

def _monitor_queue_size(queue: asyncio.Queue, queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logger.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_user(channel: Channel) -> int:
    """
        函数是代理功能的关键组件。它监听远程用户加入 Agora 频道的事件。该函数将阻塞，直到用户加入或超时。
    """
    remote_users = list(channel.remote_users.keys())
    if len(remote_users) > 0:
        return remote_users[0]

    future = asyncio.Future[int]()

    channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

    try:
        # Wait for the remote user with a timeout of 30 seconds
        remote_user = await asyncio.wait_for(future, timeout=15.0)
        return remote_user
    except KeyboardInterrupt:
        future.cancel()
        
    except Exception as e:
        logger.error(f"Error waiting for remote user: {e}")
        raise


@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    system_message: str | None = None
    turn_detection: ServerVADUpdateParams | None = None  # MARK: CHECK!
    voice: Voices | None = None


class RealtimeKitAgent:
    engine: RtcEngine
    channel: Channel
    connection: RealtimeApiConnection
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    message_queue: asyncio.Queue[ResponseAudioTranscriptDelta] = (
        asyncio.Queue()
    )
    message_done_queue: asyncio.Queue[ResponseAudioTranscriptDone] = (
        asyncio.Queue()
    )
    tools: ToolContext | None = None

    _client_tool_futures: dict[str, asyncio.Future[ClientToolCallResponse]]

    # 1.连接到Agora 和 Open Realtime API
    @classmethod
    async def setup_and_run_agent(
        cls,
        *,
        engine: RtcEngine,
        options: RtcOptions,
        inference_config: InferenceConfig,
        tools: ToolContext | None,
    ) -> None:
        """
            该setup_and_run_agent方法使用 连接到 Agora 频道RtcEngine，并使用 OpenAI 的 Realtime API 设置会话。
            它配置会话参数（例如系统消息和语音设置），并使用异步任务并发监听会话以启动和更新对话配置。
        """
        channel = engine.create_channel(options)
        await channel.connect()

        try:
            async with RealtimeApiConnection(
                base_uri=os.getenv("REALTIME_API_BASE_URI", "wss://api.openai.com"),
                api_key=os.getenv("OPENAI_API_KEY"),
                verbose=False,
            ) as connection:
                await connection.send_request(
                    SessionUpdate(
                        session=SessionUpdateParams(
                            # MARK: check this
                            turn_detection=inference_config.turn_detection,
                            tools=tools.model_description() if tools else [],
                            tool_choice="auto",
                            input_audio_format="pcm16",
                            output_audio_format="pcm16",
                            instructions=inference_config.system_message,
                            voice=inference_config.voice,
                            model=os.environ.get("OPENAI_MODEL", "gpt-4o-realtime-preview"),
                            modalities=["text", "audio"],
                            temperature=0.8,
                            max_response_output_tokens="inf",
                            input_audio_transcription=InputAudioTranscription(model="whisper-1")
                        )
                    )
                )

                start_session_message = await anext(connection.listen())
                # assert isinstance(start_session_message, messages.StartSession)
                if isinstance(start_session_message, SessionUpdated):
                    logger.info(
                        f"Session started: {start_session_message.session.id} model: {start_session_message.session.model}"
                    )
                elif isinstance(start_session_message, ErrorMessage):
                    logger.info(
                        f"Error: {start_session_message.error}"
                    )

                agent = cls(
                    connection=connection,
                    tools=tools,
                    channel=channel,
                )
                await agent.run()

        finally:
            await channel.disconnect()
            await connection.close()

    # 2.代理初始化
    def __init__(
        self,
        *,
        connection: RealtimeApiConnection,
        tools: ToolContext | None,
        channel: Channel,
    ) -> None:
        """
        设置 OpenAI 客户端、可选工具和 Agora 频道来管理实时音频通信。
        """
        self.connection = connection
        self.tools = tools
        self._client_tool_futures = {}
        self.channel = channel
        self.subscribe_user = None
        self.write_pcm = os.environ.get("WRITE_AGENT_PCM", "false") == "true"
        logger.info(f"Write PCM: {self.write_pcm}")

    async def run(self) -> None:
        """方法run是 的核心RealtimeKitAgent。它通过处理音频流、订阅远程用户以及处理传入和传出消息来管理代理的操作。此方法还确保正确的异常处理和正常关闭。
        以下是此方法的关键功能：
            等待远程用户：代理等待远程用户加入 Agora 频道并订阅他们的音频流。
            任务管理：代理启动音频输入、音频输出和处理来自 OpenAI 的消息的任务，确保它们同时运行。
            连接状态处理：监视连接状态的变化并处理用户断开连接，确保代理正常关闭。
        """
        try:

            def log_exception(t: asyncio.Task[Any]) -> None:
                if not t.cancelled() and t.exception():
                    logger.error(
                        "unhandled exception",
                        exc_info=t.exception(),
                    )

            def on_stream_message(agora_local_user, user_id, stream_id, data, length) -> None:
                logger.info(f"Received stream message with length: {length}")
            self.channel.on("stream_message", on_stream_message)

            logger.info("Waiting for remote user to join")
            self.subscribe_user = await wait_for_remote_user(self.channel)
            logger.info(f"Subscribing to user {self.subscribe_user}")
            await self.channel.subscribe_audio(self.subscribe_user)
            #await self.channel.subscribe_video(self.subscribe_user)    ###需要video时 取消注释 slg
            logger.info(f"Subscribed to audio and video from user {self.subscribe_user}")

            async def on_user_left(
                agora_rtc_conn: RTCConnection, user_id: int, reason: int
            ):
                logger.info(f"User left: {user_id}")
                if self.subscribe_user == user_id:
                    self.subscribe_user = None
                    logger.info("Subscribed user left, disconnecting")
                    await self.channel.disconnect()
            self.channel.on("user_left", on_user_left)

            disconnected_future = asyncio.Future[None]()
            def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
                logger.info(f"Connection state changed: {conn_info.state}")
                if conn_info.state == 1:
                    if not disconnected_future.done():
                        disconnected_future.set_result(None)
            self.channel.on("connection_state_changed", callback)

            asyncio.create_task(self.rtc_to_model()).add_done_callback(log_exception)
            asyncio.create_task(self.model_to_rtc()).add_done_callback(log_exception)
            asyncio.create_task(self._process_model_messages()).add_done_callback(log_exception)
            asyncio.create_task(self.tmp_save_frame()).add_done_callback(log_exception)

            await disconnected_future
            logger.info("Agent finished running")
        except asyncio.CancelledError:
            logger.info("Agent cancelled")
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

    async def rtc_to_model(self) -> None:
        # 等待直到
        while self.subscribe_user is None or self.channel.get_audio_frames(self.subscribe_user) is None:
            await asyncio.sleep(0.1)
        audio_frames = self.channel.get_audio_frames(self.subscribe_user)
        # Initialize PCMWriter for receiving audio
        pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=self.write_pcm)
        try:
            async for audio_frame in audio_frames:
                # Process received audio (send to model)
                _monitor_queue_size(self.audio_queue, "audio_queue")
                await self.connection.send_audio_data(audio_frame.data)

                # Write PCM data if enabled
                await pcm_writer.write(audio_frame.data)
                await asyncio.sleep(0)  # Yield control to allow other tasks to run

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the exception to propagate cancellation

    async def model_to_rtc(self) -> None:
        # Initialize PCMWriter for sending audio
        pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=self.write_pcm)
        try:
            while True:
                # Get audio frame from the model output
                frame = await self.audio_queue.get()

                # Process sending audio (to RTC)
                await self.channel.push_audio_frame(frame)

                # Write PCM data if enabled
                await pcm_writer.write(frame)

        except asyncio.CancelledError:
            # Write any remaining PCM data before exiting
            await pcm_writer.flush()
            raise  # Re-raise the cancelled exception to properly exit the task

    async def handle_funtion_call(self, message: ResponseFunctionCallArgumentsDone) -> None:
        function_call_response = await self.tools.execute_tool(message.name, message.arguments)
        #logger.info(f"Function call response: {function_call_response}")
        ### 返回 function call 文本信息  
        tmp_data = function_call_response.json_encoded_output
        tmp_data = tmp_data.encode('utf-8').decode('unicode_escape')
        inputdata = {"type":"function call", "transcript": tmp_data}
        asyncio.create_task(self.channel.chat.send_message(
            ChatMessage(
                message=json.dumps(inputdata), msg_id=message.item_id
            )))
        
        await self.connection.send_request(
            ItemCreate(
                item = FunctionCallOutputItemParam(
                    call_id=message.call_id,
                    output=function_call_response.json_encoded_output
                )
            )
        )
        await self.connection.send_request(
            ResponseCreate()
        )

    async def _process_model_messages(self) -> None:
        """
            处理模型返回的消息，包括音频增量和文本增量。
            消息处理是RealtimeKitAgent代理与 OpenAI 模型和 Agora 频道交互的核心。
            从模型收到的消息可以包括音频数据、文本记录或其他响应，代理需要相应地处理这些消息以【确保顺畅的实时通信】。
            实现的主要功能:
                监听消息：代理不断监听来自 OpenAI 模型的传入消息。
                处理音频数据：如果消息包含音频数据，则将其放入队列中以便在 Agora 频道播放。
                处理记录：如果消息包含部分或最终文本记录，则会对其进行处理并发送到 Agora 聊天。
                处理其他响应：根据需要处理其他消息类型，例如工具调用和其他输出。
        """
        async for message in self.connection.listen():
            # logger.info(f"Received message {message=}")
            match message:
                case ResponseAudioDelta():
                    # logger.info("Received audio message")
                    self.audio_queue.put_nowait(base64.b64decode(message.delta))
                    # loop.call_soon_threadsafe(self.audio_queue.put_nowait, base64.b64decode(message.delta))
                    logger.debug(f"TMS:ResponseAudioDelta: response_id:{message.response_id},item_id: {message.item_id}")
                case ResponseAudioTranscriptDelta():
                    # logger.info(f"Received text message {message=}")
                    # asyncio.create_task(self.channel.chat.send_message(
                    #     ChatMessage(
                    #         message=to_json(message), msg_id=message.item_id
                    #     )
                    # ))
                    pass

                case ResponseAudioTranscriptDone():
                    logger.info(f"Text message done: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))
                case InputAudioBufferSpeechStarted():
                    # 触发此事件时，系统会清除 Agora 频道上发送方的音频缓冲区并清空本地音频队列，以确保先前的音频不会干扰新的输入。它还会记录事件以进行跟踪，从而使代理能够有效地管理和处理传入的音频流。
                    await self.channel.clear_sender_audio_buffer()
                    # clear the audio queue so audio stops playing
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    logger.info(f"TMS:InputAudioBufferSpeechStarted: item_id: {message.item_id}")
                case InputAudioBufferSpeechStopped():
                    logger.info(f"TMS:InputAudioBufferSpeechStopped: item_id: {message.item_id}")
                    pass
                case ItemInputAudioTranscriptionCompleted():
                    logger.info(f"ItemInputAudioTranscriptionCompleted: {message=}")
                    asyncio.create_task(self.channel.chat.send_message(
                        ChatMessage(
                            message=to_json(message), msg_id=message.item_id
                        )
                    ))

                #  InputAudioBufferCommitted
                case InputAudioBufferCommitted():
                    pass
                case ItemCreated():
                    pass
                # ResponseCreated
                case ResponseCreated():
                    pass
                # ResponseDone
                case ResponseDone():
                    pass

                # ResponseOutputItemAdded
                case ResponseOutputItemAdded():
                    pass

                # ResponseContenPartAdded
                case ResponseContentPartAdded():
                    pass
                # ResponseAudioDone
                case ResponseAudioDone():
                    pass
                # ResponseContentPartDone
                case ResponseContentPartDone():
                    pass
                # ResponseOutputItemDone
                case ResponseOutputItemDone():
                    pass
                case SessionUpdated():
                    pass
                case RateLimitsUpdated():
                    pass
                case ResponseFunctionCallArgumentsDone():
                    asyncio.create_task(
                        self.handle_funtion_call(message)
                    )
                case ResponseFunctionCallArgumentsDelta():
                    pass

                case _:
                    logger.warning(f"Unhandled message {message=}")

    async def tmp_save_frame(self):
        while self.subscribe_user is None or self.channel.get_video_frames(self.subscribe_user) is None:
            await asyncio.sleep(0.1)
        video_frames = self.channel.get_video_frames(self.subscribe_user)
        try:
            async for video_frame in video_frames:
                # Process received video frame 
                logger.info(f"Received video frame with alpha_mode: {video_frame.alpha_mode}")
                await yuv_to_rgb(video_frame, save_path=f"frame_{video_frame.render_time_ms}.png")
                await asyncio.sleep(0)  # Yield control to allow other tasks to run
        except asyncio.CancelledError:
            raise

async def yuv_to_rgb(frame: VideoFrame, save_path: str = None) -> bytes:
    """
    将YUV格式的VideoFrame转换为RGB格式，使用PIL处理
    
    参数:
        frame: VideoFrame对象，包含YUV数据和相关参数
        save_path: 保存图片的路径，如果为None则不保存
        
    返回:
        RGB格式的字节数据
    """
    from PIL import Image
    import numpy as np
    
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
    
    # 创建PIL图像
    image = Image.fromarray(RGB, 'RGB')
    
    # 如果指定了保存路径，保存图片
    if save_path:
        try:
            image.save(save_path, quality=95)
            logger.info(f"Saved frame to {save_path}")
        except Exception as e:
            logger.error(f"Error saving frame to {save_path}: {e}")
    
    # 返回字节数据
    return RGB.tobytes()