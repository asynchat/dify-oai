"""
Dify → OpenAI 兼容 API 代理（FastAPI Router）

本模块将 Dify 应用（agent-chat / advanced-chat Chatflow）包装为 OpenAI Chat Completions
兼容接口，供 OpenAI SDK、Cursor、Dify 自定义模型等客户端直接调用。

挂载方式
--------
在 app/main.py 中通过 ``app.include_router(dify_oai_router)`` 挂载，无需单独启动服务。

对外接口
--------
- ``POST /v1/chat/completions``  对话补全（支持 stream=true/false）
- ``GET  /v1/models``            列出可用模型（模型名 = Dify 应用名称）

鉴权
----
客户端请求头::

    Authorization: Bearer <VALID_API_KEYS 中的任一 key>

``VALID_API_KEYS`` 与 ``DIFY_API_KEYS`` 分离：前者校验调用方，后者用于请求 Dify。

环境变量（.env）
----------------
必需::

    DIFY_API_BASE=http://localhost/v1      # Dify API 根路径
    DIFY_API_KEYS=app-xxx                    # Dify 应用 API Key，逗号分隔
    VALID_API_KEYS=sk-abc123                 # 对外 OpenAI 兼容鉴权 Key

可选::

    DIFY_DEFAULT_MODEL=Test                  # /info 不可用时单 Key 回退模型名
    CONVERSATION_MEMORY_MODE=1               # 会话记忆：1=history_message（默认），2=零宽字符
    DIFY_RAW_EVENT_LOG=1                     # 是否打印/落盘 Dify 原始 SSE 事件（调试）
    DIFY_RAW_EVENT_LOG_FILE=logs/dify_raw_events.jsonl

客户端扩展参数
--------------
在 OpenAI 请求体中增加（不会转发给 Dify）::

    include_tool_extensions: bool = false

- ``false``（默认）：严格 OpenAI 兼容
  - ``message.tool_calls`` 仅含 ``id / type / function.name / function.arguments``
  - 流式：``delta.tool_calls`` 增量推送 arguments
- ``true``：额外返回工具执行结果（非 OpenAI 标准）
  - 阻塞：``choices[].tool_results[]``，字段 ``tool_call_id / name / input / output``
  - 流式：``delta.tool_results[]``

OpenAI Python SDK 示例::

    from openai import OpenAI

    client = OpenAI(api_key="sk-abc123", base_url="http://localhost:8000/v1")
    resp = client.chat.completions.create(
        model="Test",                          # 与 Dify 应用名称一致
        messages=[{"role": "user", "content": "你好"}],
        extra_body={"include_tool_extensions": True},
    )

模型映射
--------
启动时调用 Dify ``/info`` 将应用名映射到 API Key。
若 ``/info`` 失败且仅配置一个 ``DIFY_API_KEYS``，则任意 model 名回退使用该 Key；
多 Key 时可设置 ``DIFY_DEFAULT_MODEL`` 或调用 ``GET /v1/models`` 查看可用名称。

阻塞 vs 流式
------------
- **流式**（``stream=true``）：忽略 ``agent_message`` 中间流（避免与 Answer 节点重复），
  转发 Answer 节点的 ``message`` 正文（含 ``<think>``）；``agent_log`` 仍转为
  ``tool_calls``；``message_end`` / ``workflow_finished`` 后立即 ``[DONE]``。
- **阻塞**（``stream=false``）：内部以 streaming 请求 Dify，收集完整回答与工具
  事件后一次性返回（advanced-chat 阻塞响应不含 agent_thoughts，必须走流式收集）。

工具调用解析
------------
从 Dify SSE 事件中提取工具 id / name / input / output，支持：

- ``agent_thought``   — agent-chat 应用
- ``agent_log``       — advanced-chat / Chatflow Agent 节点（含 CALL / Thought / ROUND）
- ``node_finished``   — 工具类工作流节点（tool / mcp / http-request 等）

调试
----
开启 ``DIFY_RAW_EVENT_LOG=1`` 后，原始事件输出到：

- 服务端终端（``./start.sh`` 窗口，非测试客户端终端）
- ``mcp-server/logs/dify_raw_events.jsonl``

本地测试::

    python tests/test_openai_client.py
"""

import json
import logging
import asyncio
import time
import base64
import tempfile
import os
import sys
import codecs

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置httpx的日志级别为WARNING，减少不必要的输出
logging.getLogger("httpx").setLevel(logging.WARNING)

# 加载环境变量
load_dotenv()

# 从环境变量读取有效的API密钥（逗号分隔）
VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]

# 获取会话记忆功能模式配置
# 1: 构造history_message附加到消息中的模式(默认)
# 2: 零宽字符模式
CONVERSATION_MEMORY_MODE = int(os.getenv('CONVERSATION_MEMORY_MODE', '1'))

# 从环境变量获取API基础URL
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")

# 是否打印 Dify 原始 SSE 事件（调试用，默认开启）
DIFY_RAW_EVENT_LOG = os.getenv("DIFY_RAW_EVENT_LOG", "1").strip().lower() in {"1", "true", "yes", "on"}

_MCP_SERVER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIFY_RAW_EVENT_LOG_FILE = os.getenv(
    "DIFY_RAW_EVENT_LOG_FILE",
    os.path.join(_MCP_SERVER_ROOT, "logs", "dify_raw_events.jsonl"),
)

RAW_DIFY_TOOL_EVENTS = {"agent_log", "agent_thought", "node_finished"}

def init_raw_event_log_file():
    """启动时创建原始事件日志文件，便于确认路径可用。"""
    if not DIFY_RAW_EVENT_LOG:
        return
    try:
        log_dir = os.path.dirname(DIFY_RAW_EVENT_LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(DIFY_RAW_EVENT_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "event": "startup",
                        "message": "Dify raw event logging enabled",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"[Dify Raw Event] log file ready: {DIFY_RAW_EVENT_LOG_FILE}", flush=True)
    except OSError as exc:
        logger.warning(f"Failed to init raw Dify event log file: {exc}")

def log_raw_dify_event(dify_chunk):
    """打印/落盘 Dify 原始 SSE 事件，便于对照真实 payload 结构。"""
    if not DIFY_RAW_EVENT_LOG:
        return
    event = dify_chunk.get("event")
    if event not in RAW_DIFY_TOOL_EVENTS:
        return

    formatted = json.dumps(dify_chunk, ensure_ascii=False, indent=2)
    message = f"[Dify Raw Event] event={event}\n{formatted}"

    # logger + print：print 会出现在 uvicorn 服务端终端
    logger.info("%s", message)
    print(message, flush=True)

    # 同时写入文件，避免用户在测试终端里找不到服务端日志
    try:
        log_dir = os.path.dirname(DIFY_RAW_EVENT_LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(DIFY_RAW_EVENT_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {"ts": time.time(), "event": event, "payload": dify_chunk},
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_file.flush()
    except OSError as exc:
        logger.warning(f"Failed to write raw Dify event log: {exc}")

class DifyModelManager:
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}  # 应用名称到API Key的映射
        self.api_key_to_name = {}  # API Key到应用名称的映射
        self.load_api_keys()

    def load_api_keys(self):
        """从环境变量加载API Keys"""
        api_keys_str = os.getenv('DIFY_API_KEYS', '')
        if api_keys_str:
            self.api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
            logger.info(f"Loaded {len(self.api_keys)} API keys")

    async def fetch_app_info(self, api_key):
        """获取Dify应用信息"""
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = await client.get(
                    f"{DIFY_API_BASE}/info",
                    headers=headers,
                    params={"user": "default_user"}
                )
                logger.info(f"Response status: {response.status_code}, content: {response.text}")
                
                if response.status_code == 200:
                    app_info = response.json()
                    return app_info.get("name", "Unknown App")
                else:
                    logger.error(f"Failed to fetch app info for API key: {api_key[:8]}...")
                    return None
        except Exception as e:
            logger.error(f"Error fetching app info: {str(e)}")
            return None

    async def refresh_model_info(self):
        """刷新所有应用信息；失败时保留已有映射并使用回退配置。"""
        new_name_to_api_key = {}
        new_api_key_to_name = {}

        for api_key in self.api_keys:
            app_name = await self.fetch_app_info(api_key)
            if app_name:
                new_name_to_api_key[app_name] = api_key
                new_api_key_to_name[api_key] = app_name
                logger.info(f"Mapped app '{app_name}' to API key: {api_key[:8]}...")

        if new_name_to_api_key:
            self.name_to_api_key = new_name_to_api_key
            self.api_key_to_name = new_api_key_to_name
            return

        if self.name_to_api_key:
            logger.warning("Dify /info unavailable, keeping existing model mappings")
            return

        self._apply_fallback_mapping()

    def _apply_fallback_mapping(self):
        """当 /info 不可用时，为单应用场景建立回退映射。"""
        if not self.api_keys:
            return

        default_model = os.getenv("DIFY_DEFAULT_MODEL", "default")
        fallback_names = [default_model]
        if default_model != "default":
            fallback_names.append("default")

        if len(self.api_keys) == 1:
            api_key = self.api_keys[0]
            for name in fallback_names:
                self.name_to_api_key[name] = api_key
            self.api_key_to_name[api_key] = default_model
            logger.warning(
                f"Dify /info unavailable, fallback to single API key with model names: "
                f"{', '.join(fallback_names)}"
            )
            return

        for index, api_key in enumerate(self.api_keys):
            model_name = f"dify-app-{index + 1}"
            self.name_to_api_key[model_name] = api_key
            self.api_key_to_name[api_key] = model_name
        logger.warning(
            "Dify /info unavailable, fallback to indexed model names: "
            f"{', '.join(self.name_to_api_key.keys())}"
        )

    def get_api_key(self, model_name):
        """根据模型名称获取API Key"""
        api_key = self.name_to_api_key.get(model_name)
        if api_key:
            return api_key

        if len(self.api_keys) == 1:
            return self.api_keys[0]

        return None

    async def resolve_api_key(self, model_name):
        """解析模型对应的 API Key，必要时刷新映射。"""
        api_key = self.get_api_key(model_name)
        if api_key:
            return api_key

        await self.refresh_model_info()
        api_key = self.get_api_key(model_name)
        if api_key:
            return api_key

        if len(self.api_keys) == 1:
            logger.warning(
                f"Model '{model_name}' not mapped, using the only configured Dify API key"
            )
            return self.api_keys[0]

        return None

    def get_available_models(self):
        """获取可用模型列表"""
        if not self.name_to_api_key and self.api_keys:
            self._apply_fallback_mapping()

        models = [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify"
            }
            for name in self.name_to_api_key.keys()
        ]

        if not models and len(self.api_keys) == 1:
            models.append({
                "id": os.getenv("DIFY_DEFAULT_MODEL", "default"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify",
            })

        return models

# 创建模型管理器实例
model_manager = DifyModelManager()

router = APIRouter()

async def resolve_api_key(model_name):
    """根据模型名称获取对应的 Dify API 密钥。"""
    api_key = await model_manager.resolve_api_key(model_name)
    if not api_key:
        logger.warning(f"No API key found for model: {model_name}")
    return api_key

async def upload_image_to_dify(api_key, base64_data, user_id="default_user"):
    """上传图片到Dify并返回文件ID
    支持处理base64编码的图片数据，自动检测并提取有效的base64数据
    """
    try:
        # 解码base64数据
        if base64_data.startswith('data:image'):
            # 提取实际的base64数据 (去除data:image/*;base64,前缀)
            base64_data = base64_data.split(',')[1]
        
        image_data = base64.b64decode(base64_data)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_file_path = tmp_file.name
        
        try:
            # 使用httpx上传文件到Dify
            async with httpx.AsyncClient(timeout=None) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}"
                }
                
                # 准备multipart数据用于文件上传
                # Dify当前仅支持图片类型附件的上传 (PNG, JPG, JPEG, WEBP, GIF)
                with open(tmp_file_path, 'rb') as file_handle:
                    files = {
                        'file': ('image.png', file_handle, 'image/png')
                    }
                    data = {
                        'user': user_id
                    }
                    
                    response = await client.post(
                        f"{DIFY_API_BASE}/files/upload",
                        headers=headers,
                        files=files,
                        data=data
                    )
                
                # 检查上传响应状态码
                # HTTP 200: OK, HTTP 201: Created
                if response.status_code in [200, 201]:
                    file_info = response.json()
                    logger.info(f"Successfully uploaded image, file_id: {file_info.get('id')}")
                    return file_info.get('id')
                else:
                    logger.error(f"Failed to upload image, status_code: {response.status_code}, response: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return None
            
        finally:
            # 确保临时文件被清理，避免磁盘空间泄露
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    # 等待一小段时间确保文件句柄完全释放
                    await asyncio.sleep(0.1)
                    os.unlink(tmp_file_path)
                    logger.debug(f"Temporary file cleaned up: {tmp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {tmp_file_path}: {cleanup_error}")
                    # 如果立即删除失败，尝试延迟删除
                    try:
                        await asyncio.sleep(1)
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                            logger.debug(f"Temporary file cleaned up after delay: {tmp_file_path}")
                    except Exception as delayed_cleanup_error:
                        logger.error(f"Failed to cleanup temporary file after delay {tmp_file_path}: {delayed_cleanup_error}")
            
    except Exception as e:
        logger.error(f"Error processing image data: {str(e)}")
        return None

async def transform_openai_to_dify(openai_request, endpoint, api_key=None):
    """将OpenAI格式的请求转换为Dify格式"""
    
    if endpoint == "/chat/completions":
        messages = openai_request.get("messages", [])
        stream = openai_request.get("stream", False)
        user_id = openai_request.get("user", "default_user")
        inputs = openai_request.get("inputs", {})
        
        # 尝试从历史消息中提取conversation_id
        conversation_id = None
        
        # 提取system消息内容
        system_content = ""
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        if system_messages:
            system_content = system_messages[0].get("content", "")
            # 记录找到的system消息
            logger.info(f"Found system message: {system_content[:100]}{'...' if len(system_content) > 100 else ''}")
        
        # 处理用户消息，支持图片
        user_message = messages[-1] if messages and messages[-1].get("role") != "system" else {}
        user_content = user_message.get("content", "")
        
        # 存储上传的文件ID
        uploaded_files = []
        
        # 检查用户消息是否包含图片
        if isinstance(user_content, list):
            # 处理多模态内容（文本+图片）
            text_parts = []
            image_parts = []
            
            for item in user_content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        image_parts.append(image_url)
            
            # 组合文本内容
            user_query = "\n".join(text_parts) if text_parts else ""
            
            # 上传图片文件
            if api_key and image_parts:
                logger.info(f"Found {len(image_parts)} images to upload")
                successful_uploads = 0
                failed_uploads = 0
                
                for i, image_data in enumerate(image_parts):
                    try:
                        logger.info(f"Uploading image {i+1}/{len(image_parts)}")
                        file_id = await upload_image_to_dify(api_key, image_data, user_id)
                        if file_id:
                            uploaded_files.append({
                                "type": "image",
                                "transfer_method": "local_file",
                                "upload_file_id": file_id
                            })
                            successful_uploads += 1
                            logger.info(f"Successfully uploaded image {i+1}/{len(image_parts)}, file_id: {file_id}")
                        else:
                            failed_uploads += 1
                            logger.warning(f"Failed to upload image {i+1}/{len(image_parts)}")
                    except Exception as e:
                        failed_uploads += 1
                        logger.error(f"Exception occurred while uploading image {i+1}/{len(image_parts)}: {str(e)}")
                
                # 记录上传结果统计
                if successful_uploads > 0:
                    logger.info(f"Uploaded {successful_uploads}/{len(image_parts)} files successfully")
                if failed_uploads > 0:
                    logger.warning(f"Failed to upload {failed_uploads}/{len(image_parts)} files")
                
                # 如果所有图片都上传失败，记录警告
                if successful_uploads == 0 and failed_uploads > 0:
                    logger.warning("All image uploads failed, proceeding with text-only request")
        else:
            # 处理纯文本内容
            user_query = user_content
        
        logger.info(f"Processing request with {len(uploaded_files)} uploaded files")
        
        if CONVERSATION_MEMORY_MODE == 2:  # 零宽字符模式
            if len(messages) > 1:
                # 遍历历史消息，找到最近的assistant消息
                for msg in reversed(messages[:-1]):  # 除了最后一条消息
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # 尝试解码conversation_id
                        conversation_id = decode_conversation_id(content)
                        if conversation_id:
                            break
            
            # 如果有system消息且是首次对话(没有conversation_id)，则将system内容添加到用户查询前
            if system_content and not conversation_id:
                user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"
                logger.info(f"[零宽字符模式] 首次对话，添加system内容到查询前")
            
            dify_request = {
                "inputs": inputs,
                "query": user_query,
                "response_mode": "streaming" if stream else "blocking",
                "conversation_id": conversation_id,
                "user": user_id
            }
            
            # 如果有上传的文件，添加到请求中
            if uploaded_files:
                dify_request["files"] = uploaded_files
                
        else:  # history_message模式(默认)
            # 构造历史消息
            if len(messages) > 1:
                history_messages = []
                has_system_in_history = False
                
                # 检查历史消息中是否已经包含system消息
                for msg in messages[:-1]:  # 除了最后一条消息
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        if role == "system":
                            has_system_in_history = True
                        history_messages.append(f"{role}: {content}")
                
                # 如果历史中没有system消息但现在有system消息，则添加到历史的最前面
                if system_content and not has_system_in_history:
                    history_messages.insert(0, f"system: {system_content}")
                    logger.info(f"[history_message模式] 添加system内容到历史消息前")
                
                # 将历史消息添加到查询中
                if history_messages:
                    history_context = "\n\n".join(history_messages)
                    user_query = f"<history>\n{history_context}\n</history>\n\n用户当前问题: {user_query}"
            elif system_content:  # 没有历史消息但有system消息
                user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"
                logger.info(f"[history_message模式] 首次对话，添加system内容到查询前")
            
            dify_request = {
                "inputs": inputs,
                "query": user_query,
                "response_mode": "streaming" if stream else "blocking",
                "user": user_id
            }
            
            # 如果有上传的文件，添加到请求中
            if uploaded_files:
                dify_request["files"] = uploaded_files

        return dify_request
    
    return None

def _normalize_tool_field(value):
    """将工具字段统一转为字符串。"""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def parse_include_tool_extensions(openai_request):
    """解析客户端是否请求输出工具扩展字段，默认关闭（严格 OpenAI 模式）。"""
    value = openai_request.get("include_tool_extensions", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _pick_first(mapping, *keys):
    """从字典中按优先级取第一个非空字段。"""
    if not isinstance(mapping, dict):
        return ""
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return ""

def _as_json_value(value):
    """解析 JSON 字符串或直接返回 dict/list 原值。"""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return None

def _as_mapping(value):
    parsed = _as_json_value(value)
    if isinstance(parsed, dict):
        return parsed
    return {}

def _is_thought_like_content(text):
    """判断内容是否像 LLM 思考过程，而非工具返回值。"""
    if not text:
        return False
    normalized = str(text).strip().lower()
    thought_markers = (
        "<think>",
        "the user said",
        "let me call the tool",
        "let me analyze",
        "according to the rules",
        "according to my instructions",
        "i should use the default",
        "i need to",
    )
    return any(marker in normalized for marker in thought_markers)

def should_forward_dify_content_chunk(dify_chunk):
    """是否将 Dify SSE 片段作为 OpenAI content 转发（忽略 agent_message 中间思考流）。"""
    event = dify_chunk.get("event")
    if event == "agent_message":
        return False
    return event in {"message", "message_replace"}

def is_dify_stream_terminal_event(dify_chunk):
    """Dify 流式会话结束事件。"""
    return dify_chunk.get("event") in {"message_end", "workflow_finished"}

def _is_valid_tool_agent_log(label, parsed_inner, metadata):
    """过滤 agent_log 中非工具调用/工具结果的事件。"""
    if isinstance(parsed_inner, dict):
        if _pick_first(parsed_inner, "tool_name", "tool_input", "tool_call_name"):
            return True
        output_block = parsed_inner.get("output")
        if isinstance(output_block, dict) and (
            output_block.get("tool_call_name")
            or output_block.get("tool_responses")
        ):
            return True

    label_lower = (label or "").strip().lower()
    if label_lower.startswith("call "):
        return True
    if label_lower.startswith("round ") and isinstance(parsed_inner, dict):
        output_block = parsed_inner.get("output")
        if isinstance(output_block, dict) and output_block.get("tool_responses"):
            return True

    return False

def _iter_tool_records_from_agent_log_inner(parsed_inner, label=""):
    """从 agent_log.data.data 中迭代提取工具记录 (name, input, output, tool_call_id)。"""
    if not isinstance(parsed_inner, dict):
        return

    tool_name = _pick_first(parsed_inner, "tool_name", "tool", "tool_call_name")
    tool_input = parsed_inner.get("tool_input")
    output_block = parsed_inner.get("output")

    label_lower = (label or "").strip().lower()
    if label_lower.startswith("call "):
        tool_name = tool_name or label[5:].strip()

    if isinstance(output_block, dict):
        if output_block.get("tool_call_name") or output_block.get("tool_call_input") or output_block.get("tool_response"):
            yield (
                output_block.get("tool_call_name") or tool_name,
                output_block.get("tool_call_input") or tool_input,
                output_block.get("tool_response") or "",
                output_block.get("tool_call_id"),
            )
        tool_responses = output_block.get("tool_responses")
        if isinstance(tool_responses, list):
            for item in tool_responses:
                if not isinstance(item, dict):
                    continue
                yield (
                    item.get("tool_call_name") or tool_name,
                    item.get("tool_call_input") or tool_input,
                    item.get("tool_response") or "",
                    item.get("tool_call_id"),
                )
        return

    if tool_name and tool_input:
        yield tool_name, tool_input, "", None

def _normalize_tool_input_value(value, tool_name=""):
    """规范化工具 input，兼容 Dify agent_log 的数组格式。"""
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            item = value[0]
            if tool_name and item.get("name") == tool_name and isinstance(item.get("args"), dict):
                return item["args"]
            if isinstance(item.get("args"), dict):
                return item["args"]
        return value
    if isinstance(value, dict):
        if tool_name and tool_name in value and len(value) == 1:
            return value[tool_name]
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return _normalize_tool_input_value(parsed, tool_name)
        except json.JSONDecodeError:
            return value
    return value

def _extract_tool_output_from_mapping(*mappings):
    """优先提取真实工具返回值，跳过 thinking 内容。"""
    output_keys = (
        "tool_response",
        "tool_output",
        "observation",
        "result",
        "response",
    )
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        if isinstance(mapping.get("tool_call"), dict):
            nested = _extract_tool_output_from_mapping(mapping["tool_call"])
            if nested:
                return nested
        output_block = mapping.get("output")
        if isinstance(output_block, dict):
            nested = _extract_tool_output_from_mapping(output_block)
            if nested:
                return nested
            tool_responses = output_block.get("tool_responses")
            if isinstance(tool_responses, list):
                for item in tool_responses:
                    nested = _extract_tool_output_from_mapping(item)
                    if nested:
                        return nested
        for key in output_keys:
            value = mapping.get(key)
            if value is None or value == "":
                continue
            normalized = _normalize_tool_field(value)
            if normalized and not _is_thought_like_content(normalized):
                return value
        generic_output = mapping.get("output")
        if isinstance(generic_output, str):
            normalized = _normalize_tool_field(generic_output)
            if normalized and not _is_thought_like_content(normalized):
                return generic_output
    return ""

def _has_tool_payload(value):
    if value is None or value == "" or value == {} or value == []:
        return False
    return True

def _extract_tool_input_from_sources(tool_name, *sources):
    """从 dict/list/JSON 字符串中提取工具 input。"""
    for source in sources:
        if source is None or source == "":
            continue
        parsed = _as_json_value(source)
        if isinstance(parsed, list):
            normalized = _normalize_tool_input_value(parsed, tool_name)
            if _has_tool_payload(normalized):
                return normalized
        if isinstance(parsed, dict):
            nested = _extract_tool_input_from_mapping(tool_name, parsed)
            if _has_tool_payload(nested):
                return nested
        if isinstance(source, dict):
            nested = _extract_tool_input_from_mapping(tool_name, source)
            if _has_tool_payload(nested):
                return nested
    return ""

def _extract_tool_input_from_mapping(tool_name, *mappings):
    """从 mapping 字段中提取工具 input。"""
    input_keys = (
        "tool_input",
        "tool_call_input",
        "input",
        "arguments",
        "params",
        "tool_params",
        "args",
    )
    for mapping in mappings:
        if isinstance(mapping, list):
            normalized = _normalize_tool_input_value(mapping, tool_name)
            if _has_tool_payload(normalized):
                return normalized
        if not isinstance(mapping, dict):
            continue
        if isinstance(mapping.get("tool_call"), dict):
            nested = _extract_tool_input_from_mapping(tool_name, mapping["tool_call"])
            if _has_tool_payload(nested):
                return nested
        for key in input_keys:
            value = mapping.get(key)
            if value is None or value == "":
                continue
            normalized = _normalize_tool_input_value(value, tool_name)
            if _has_tool_payload(normalized):
                return normalized
    return ""

def merge_tool_execution_record(existing, new):
    """合并同一工具调用的多次 agent_log 更新，避免 thinking 覆盖真实 output。"""
    merged_name = new.get("name") or existing.get("name", "")
    merged_input = new.get("input") if new.get("input") else existing.get("input", "")

    existing_output = existing.get("output", "")
    new_output = new.get("output", "")
    if _is_thought_like_content(new_output) and existing_output and not _is_thought_like_content(existing_output):
        merged_output = existing_output
    elif new_output and not _is_thought_like_content(new_output):
        merged_output = new_output
    else:
        merged_output = existing_output or new_output

    merged = {
        "id": existing.get("id") or new.get("id"),
        "name": merged_name,
        "input": merged_input,
        "output": merged_output,
    }
    stream_key = existing.get("_stream_key") or new.get("_stream_key")
    if stream_key:
        merged["_stream_key"] = stream_key
    return merged

def store_tool_execution(tool_store, execution, merge_key=None):
    """写入工具执行记录，支持按 merge_key 合并。"""
    key = merge_key or execution["id"]
    if key in tool_store:
        tool_store[key] = merge_tool_execution_record(tool_store[key], execution)
    else:
        tool_store[key] = execution

def update_tool_execution_snapshot(snapshots, execution):
    """更新流式工具快照，返回 (merged_execution, changed)。"""
    merge_key = execution.pop("_merge_key", None)
    key = merge_key or execution["id"]
    previous = snapshots.get(key)
    merged = merge_tool_execution_record(previous, execution) if previous else execution
    merged["_stream_key"] = key
    changed = previous != merged
    snapshots[key] = merged
    return merged, changed

TOOL_NODE_TYPES = {
    "tool",
    "http-request",
    "code",
    "mcp",
    "tool-provider",
}

def build_tool_execution(thought_id, tool_name, tool_input, observation):
    """从 Dify 事件提取工具执行信息。"""
    name = _normalize_tool_field(tool_name)
    normalized_input = _normalize_tool_input_value(tool_input, name)
    input_value = _normalize_tool_field(normalized_input)
    output_value = _normalize_tool_field(observation)

    if output_value and _is_thought_like_content(output_value):
        output_value = ""

    if not any([name, input_value, output_value]):
        return None

    return {
        "id": str(thought_id) if thought_id else f"call_{int(time.time() * 1000)}",
        "name": name,
        "input": input_value,
        "output": output_value,
    }

def build_tool_executions_from_agent_thought(dify_chunk):
    """从 agent_thought 事件构建工具执行信息（支持多工具）。"""
    tool_field = dify_chunk.get("tool") or ""
    tools = [item.strip() for item in tool_field.split(";") if item.strip()]
    tool_input_raw = dify_chunk.get("tool_input", "")
    observation = dify_chunk.get("observation", "")
    thought_id = dify_chunk.get("id", "")

    tool_inputs = _as_json_value(tool_input_raw)
    if not isinstance(tool_inputs, dict):
        tool_inputs = {}
    if not tool_inputs and tool_input_raw:
        tool_inputs = {"": tool_input_raw}

    if not tools:
        direct_input = _extract_tool_input_from_sources("", tool_input_raw)
        execution = build_tool_execution(thought_id, "", direct_input or tool_input_raw, observation)
        return [execution] if execution else []

    executions = []
    for index, tool_name in enumerate(tools):
        exec_id = f"{thought_id}_{index}" if len(tools) > 1 else thought_id
        if isinstance(tool_inputs, dict) and tool_name in tool_inputs:
            input_value = tool_inputs.get(tool_name)
        else:
            input_value = _extract_tool_input_from_sources(tool_name, tool_input_raw)
            if not input_value and len(tools) == 1:
                input_value = tool_input_raw
        output = observation if len(tools) == 1 else ""
        execution = build_tool_execution(exec_id, tool_name, input_value, output)
        if execution:
            executions.append(execution)
    return executions

def build_tool_executions_from_agent_log(dify_chunk):
    """从 advanced-chat / Chatflow 的 agent_log 事件构建工具执行信息。"""
    payload = dify_chunk.get("data") or {}
    raw_inner = payload.get("data")
    parsed_inner = _as_json_value(raw_inner)
    if not isinstance(parsed_inner, dict):
        parsed_inner = {}
    metadata = _as_mapping(payload.get("metadata"))
    label = (payload.get("label") or "").strip()

    if not _is_valid_tool_agent_log(label, parsed_inner, metadata):
        logger.info(f"[Agent Log] skipped non-tool log label={label!r}")
        return []

    executions = []
    seen_keys = set()
    for tool_name, tool_input, tool_output, tool_call_id in _iter_tool_records_from_agent_log_inner(parsed_inner, label):
        execution_id = tool_call_id or payload.get("id") or payload.get("node_execution_id") or dify_chunk.get("id")
        execution = build_tool_execution(execution_id, tool_name, tool_input, tool_output)
        if not execution:
            continue

        merge_key = (
            f"{payload.get('node_execution_id')}:{tool_name}"
            if payload.get("node_execution_id") and tool_name
            else (tool_call_id or execution["id"])
        )
        if merge_key in seen_keys:
            continue
        seen_keys.add(merge_key)
        execution["_merge_key"] = merge_key
        executions.append(execution)

    if executions:
        first = executions[0]
        logger.info(
            f"[Agent Log] label={label!r}, tool={first['name']}, "
            f"input={first['input'][:120] if first['input'] else ''}, "
            f"has_output={bool(first['output'])}"
        )
    elif label:
        logger.info(f"[Agent Log] unparsed tool log label={label!r}, inner={json.dumps(parsed_inner, ensure_ascii=False)[:500]}")

    return executions

def build_tool_executions_from_node_finished(dify_chunk):
    """从 node_finished 事件构建工具执行信息。"""
    payload = dify_chunk.get("data") or {}
    node_type = (payload.get("node_type") or "").lower()

    if node_type == "agent":
        return []

    if node_type not in TOOL_NODE_TYPES and "tool" not in node_type:
        return []

    tool_name = payload.get("title") or payload.get("node_id")
    tool_input = payload.get("inputs", {})
    output = payload.get("outputs", {})
    execution = build_tool_execution(payload.get("id"), tool_name, tool_input, output)
    return [execution] if execution else []

def extract_tool_executions_from_dify_event(dify_chunk):
    """从 Dify SSE 事件中提取工具执行记录。"""
    event = dify_chunk.get("event")
    if event == "agent_thought":
        return build_tool_executions_from_agent_thought(dify_chunk)
    if event == "agent_log":
        executions = build_tool_executions_from_agent_log(dify_chunk)
        if executions:
            logger.info(
                f"[Agent Log] tool={executions[0]['name']}, "
                f"has_input={bool(executions[0]['input'])}, has_output={bool(executions[0]['output'])}"
            )
        return executions
    if event == "node_finished":
        executions = build_tool_executions_from_node_finished(dify_chunk)
        if executions:
            logger.info(
                f"[Node Finished] node_type={dify_chunk.get('data', {}).get('node_type')}, "
                f"tool={executions[0]['name']}"
            )
        return executions
    return []

def extract_tool_executions_from_dify_response(dify_response):
    """从 Dify 阻塞响应中提取全部工具执行记录。"""
    executions = []
    seen_ids = set()

    candidate_thoughts = list(dify_response.get("agent_thoughts", []))
    metadata = dify_response.get("metadata") or {}
    if isinstance(metadata, dict):
        candidate_thoughts.extend(metadata.get("agent_thoughts", []))

    for thought in candidate_thoughts:
        for execution in build_tool_executions_from_agent_thought(thought):
            if execution["id"] not in seen_ids:
                seen_ids.add(execution["id"])
                executions.append(execution)

    return executions

class SseLineReader:
    """增量解码 SSE 字节流，避免 UTF-8 多字节字符在 chunk 边界被截断。"""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._text_buffer = ""

    def feed(self, chunk: bytes) -> list[str]:
        if not chunk:
            return []
        self._text_buffer += self._decoder.decode(chunk)
        lines = []
        while "\n" in self._text_buffer:
            line, self._text_buffer = self._text_buffer.split("\n", 1)
            lines.append(line)
        return lines

    def flush(self) -> list[str]:
        remainder = self._decoder.decode(b"", final=True)
        if remainder:
            self._text_buffer += remainder
        if not self._text_buffer:
            return []
        line = self._text_buffer
        self._text_buffer = ""
        return [line]

def parse_sse_data_lines(lines):
    """从 SSE 文本行中解析 JSON 事件。"""
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        try:
            chunk = json.loads(line[6:])
            log_raw_dify_event(chunk)
            yield chunk
        except json.JSONDecodeError as exc:
            logger.error(f"JSON decode error: {exc}")

async def iter_dify_sse_chunks(client, dify_endpoint, dify_request, headers):
    """迭代 Dify SSE 事件。"""
    stream_request = {**dify_request, "response_mode": "streaming"}
    async with client.stream(
        "POST",
        dify_endpoint,
        json=stream_request,
        headers={
            **headers,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    ) as response:
        if response.status_code != 200:
            error_body = (await response.aread()).decode("utf-8", errors="replace")
            raise httpx.HTTPStatusError(
                f"Dify API error: {error_body}",
                request=response.request,
                response=response,
            )

        line_reader = SseLineReader()
        async for raw_bytes in response.aiter_raw():
            for line in line_reader.feed(raw_bytes):
                for chunk in parse_sse_data_lines([line]):
                    yield chunk

        for line in line_reader.flush():
            for chunk in parse_sse_data_lines([line]):
                yield chunk

async def collect_dify_response_via_stream(client, dify_endpoint, dify_request, headers):
    """通过内部流式请求收集完整回答与工具执行信息（兼容 advanced-chat）。"""
    answer_parts = []
    tool_executions = {}
    message_id = None
    conversation_id = None

    async for dify_chunk in iter_dify_sse_chunks(client, dify_endpoint, dify_request, headers):
        event = dify_chunk.get("event")
        if should_forward_dify_content_chunk(dify_chunk) and dify_chunk.get("answer") is not None:
            if event == "message_replace":
                answer_parts = [dify_chunk["answer"]]
            else:
                answer_parts.append(dify_chunk["answer"])
        if dify_chunk.get("message_id"):
            message_id = dify_chunk.get("message_id")
        if dify_chunk.get("conversation_id"):
            conversation_id = dify_chunk.get("conversation_id")
        if dify_chunk.get("id") and not message_id and event in {"message", "agent_message", "message_end"}:
            message_id = dify_chunk.get("id")

        for execution in extract_tool_executions_from_dify_event(dify_chunk):
            merge_key = execution.pop("_merge_key", execution["id"])
            store_tool_execution(tool_executions, execution, merge_key)

    return {
        "answer": "".join(answer_parts),
        "tool_executions": list(tool_executions.values()),
        "message_id": message_id or "",
        "conversation_id": conversation_id or "",
    }

def format_strict_tool_call(execution):
    """严格 OpenAI 格式的 tool_call（仅 id / type / function）。"""
    return {
        "id": execution["id"],
        "type": "function",
        "function": {
            "name": execution["name"],
            "arguments": execution["input"],
        },
    }

def format_tool_result_extension(execution):
    """扩展字段：工具执行结果（非 OpenAI 标准，需客户端显式开启）。"""
    return {
        "tool_call_id": execution["id"],
        "name": execution["name"],
        "input": execution["input"],
        "output": execution["output"],
    }

def split_arguments_for_stream(arguments, chunk_size=16):
    """将 arguments 拆分为流式片段，模拟 OpenAI 增量推送。"""
    if not arguments:
        return [""]
    return [arguments[i:i + chunk_size] for i in range(0, len(arguments), chunk_size)]

def create_strict_tool_call_stream_chunks(message_id, model, execution, index):
    """生成严格 OpenAI 兼容的 tool_calls 流式 chunks。"""
    chunks = []

    chunks.append({
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": index,
                    "id": execution["id"],
                    "type": "function",
                    "function": {
                        "name": execution["name"],
                        "arguments": "",
                    },
                }]
            },
            "finish_reason": None,
        }]
    })

    for argument_part in split_arguments_for_stream(execution["input"]):
        if not argument_part:
            continue
        chunks.append({
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": index,
                        "function": {
                            "arguments": argument_part,
                        },
                    }]
                },
                "finish_reason": None,
            }]
        })

    return chunks

def create_tool_result_extension_chunk(message_id, model, execution):
    """生成工具结果扩展字段的流式 chunk（非 OpenAI 标准）。"""
    return {
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_results": [format_tool_result_extension(execution)],
            },
            "finish_reason": None,
        }]
    }

def transform_dify_to_openai(
    dify_response,
    model="claude-3-5-sonnet-v2",
    stream=False,
    include_tool_extensions=False,
    tool_executions=None,
    answer_override=None,
):
    """将Dify格式的响应转换为OpenAI格式"""
    
    if not stream:
        # 首先获取回答内容，支持不同的响应模式
        answer = answer_override if answer_override is not None else ""
        mode = dify_response.get("mode", "")
        
        if answer_override is None:
            # 普通聊天模式
            if "answer" in dify_response:
                answer = dify_response.get("answer", "")
            
            # 如果是Agent模式，需要从agent_thoughts中提取回答
            elif "agent_thoughts" in dify_response:
                # Agent模式下通常最后一个thought包含最终答案
                agent_thoughts = dify_response.get("agent_thoughts", [])
                if agent_thoughts:
                    for thought in agent_thoughts:
                        if thought.get("thought"):
                            answer = thought.get("thought", "")

        # 只在零宽字符会话记忆模式时处理conversation_id
        if CONVERSATION_MEMORY_MODE == 2:
            conversation_id = dify_response.get("conversation_id", "")
            history = dify_response.get("conversation_history", [])
            
            # 检查历史消息中是否已经有会话ID
            has_conversation_id = False
            if history:
                for msg in history:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if decode_conversation_id(content) is not None:
                            has_conversation_id = True
                            break
            
            # 只在新会话且历史消息中没有会话ID时插入
            if conversation_id and not has_conversation_id:
                logger.info(f"[Debug] Inserting conversation_id: {conversation_id}, history_length: {len(history)}")
                encoded = encode_conversation_id(conversation_id)
                answer = answer + encoded
                logger.info(f"[Debug] Response content after insertion: {repr(answer)}")

        if tool_executions is None:
            tool_executions = extract_tool_executions_from_dify_response(dify_response)
        tool_calls = [format_strict_tool_call(item) for item in tool_executions]

        message = {
            "role": "assistant",
            "content": answer if answer else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        finish_reason = "tool_calls" if tool_calls and not answer else "stop"

        choice = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }
        if include_tool_extensions and tool_executions:
            choice["tool_results"] = [
                format_tool_result_extension(item) for item in tool_executions
            ]

        return {
            "id": dify_response.get("message_id", "") if isinstance(dify_response, dict) else "",
            "object": "chat.completion",
            "created": dify_response.get("created", int(time.time())) if isinstance(dify_response, dict) else int(time.time()),
            "model": model,
            "choices": [choice],
        }
    else:
        # 流式响应的转换在stream_response函数中处理
        return dify_response

def create_openai_stream_response(content, message_id, model="claude-3-5-sonnet-v2"):
    """创建OpenAI格式的流式响应"""
    return {
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": content
            },
            "finish_reason": None
        }]
    }

def encode_conversation_id(conversation_id):
    """将conversation_id编码为不可见的字符序列"""
    if not conversation_id:
        return ""
    
    encoded = base64.b64encode(conversation_id.encode()).decode()
    
    char_map = {
        '0': '\u200b',
        '1': '\u200c',
        '2': '\u200d',
        '3': '\ufeff',
        '4': '\u2060',
        '5': '\u180e',
        '6': '\u2061',
        '7': '\u2062',
    }
    
    result = []
    for c in encoded:
        if c.isalpha():
            if c.isupper():
                val = ord(c) - ord('A')
            else:
                val = ord(c) - ord('a') + 26
        elif c.isdigit():
            val = int(c) + 52
        elif c == '+':
            val = 62
        elif c == '/':
            val = 63
        else:  # '='
            val = 0
            
        first = (val >> 3) & 0x7
        second = val & 0x7
        result.append(char_map[str(first)])
        if c != '=':
            result.append(char_map[str(second)])
    
    return ''.join(result)

def decode_conversation_id(content):
    """从消息内容中解码conversation_id"""
    try:
        char_to_val = {
            '\u200b': '0',
            '\u200c': '1',
            '\u200d': '2',
            '\ufeff': '3',
            '\u2060': '4',
            '\u180e': '5',
            '\u2061': '6',
            '\u2062': '7',
        }
        
        space_chars = []
        for c in reversed(content):
            if c not in char_to_val:
                break
            space_chars.append(c)
        
        if not space_chars:
            return None
            
        space_chars.reverse()
        base64_chars = []
        for i in range(0, len(space_chars), 2):
            first = int(char_to_val[space_chars[i]], 8)
            if i + 1 < len(space_chars):
                second = int(char_to_val[space_chars[i + 1]], 8)
                val = (first << 3) | second
            else:
                val = first << 3
                
            if val < 26:
                base64_chars.append(chr(val + ord('A')))
            elif val < 52:
                base64_chars.append(chr(val - 26 + ord('a')))
            elif val < 62:
                base64_chars.append(str(val - 52))
            elif val == 62:
                base64_chars.append('+')
            else:
                base64_chars.append('/')
                
        padding = len(base64_chars) % 4
        if padding:
            base64_chars.extend(['='] * (4 - padding))
            
        base64_str = ''.join(base64_chars)
        return base64.b64decode(base64_str).decode()
        
    except Exception as e:
        logger.debug(f"Failed to decode conversation_id: {e}")
        return None

@router.post('/v1/chat/completions')
async def chat_completions(request: Request):
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return JSONResponse(status_code=401, content={
                "error": {
                    "message": "Missing Authorization header",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            })

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return JSONResponse(status_code=401, content={
                "error": {
                    "message": "Invalid Authorization header format. Expected: Bearer <API_KEY>",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            })

        provided_api_key = parts[1]
        if provided_api_key not in VALID_API_KEYS:
            return JSONResponse(status_code=401, content={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            })

        openai_request = await request.json()
        include_tool_extensions = parse_include_tool_extensions(openai_request)
        
        logger.info(
            f"Received request: {json.dumps(openai_request, ensure_ascii=False)}, "
            f"include_tool_extensions={include_tool_extensions}"
        )
        
        model = openai_request.get("model", "claude-3-5-sonnet")
        
        api_key = await resolve_api_key(model)
        if not api_key:
            available = model_manager.get_available_models()
            available_names = [item["id"] for item in available]
            error_msg = (
                f"Model {model} is not supported. "
                f"Available models: {', '.join(available_names) if available_names else '(none)'}"
            )
            logger.error(error_msg)
            return JSONResponse(status_code=404, content={
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })
            
        dify_request = await transform_openai_to_dify(openai_request, "/chat/completions", api_key)
        
        if '--debug' in sys.argv:
            print("=" * 50)
            print("TRANSFORMED REQUEST DEBUG INFO")
            print("=" * 50)
            print(f"Transformed Body: {json.dumps(dify_request, ensure_ascii=False, indent=2)}")
            print("=" * 50)
        
        if not dify_request:
            logger.error("Failed to transform request")
            return JSONResponse(status_code=400, content={
                "error": {
                    "message": "Invalid request format",
                    "type": "invalid_request_error",
                }
            })

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stream = openai_request.get("stream", False)
        dify_endpoint = f"{DIFY_API_BASE}/chat-messages"
        logger.info(f"Sending request to Dify endpoint: {dify_endpoint}, stream={stream}")

        if stream:
            async def generate():
                async with httpx.AsyncClient(timeout=None) as client:
                    def send_content_delta(content, message_id):
                        openai_chunk = {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }],
                        }
                        return f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

                    def send_sse_chunk(chunk_data):
                        return f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

                    msg_state_id = None
                    stream_done = False
                    tool_call_index = {}
                    tool_call_count = 0
                    tool_execution_snapshots = {}
                    emitted_tool_call_headers = set()
                    emitted_arguments_length = {}
                    emitted_tool_result_output = {}

                    def emit_tool_execution_events(execution):
                        nonlocal tool_call_count
                        tool_id = execution.get("_stream_key") or execution["id"]
                        if tool_id not in tool_call_index:
                            tool_call_index[tool_id] = tool_call_count
                            tool_call_count += 1
                        index = tool_call_index[tool_id]

                        if tool_id not in emitted_tool_call_headers:
                            emitted_tool_call_headers.add(tool_id)
                            header_chunk = create_strict_tool_call_stream_chunks(
                                msg_state_id, model, execution, index
                            )[0]
                            yield send_sse_chunk(header_chunk)
                            emitted_arguments_length[tool_id] = 0

                        current_args = execution["input"]
                        previous_len = emitted_arguments_length.get(tool_id, 0)
                        if len(current_args) > previous_len:
                            new_part = current_args[previous_len:]
                            emitted_arguments_length[tool_id] = len(current_args)
                            for argument_part in split_arguments_for_stream(new_part):
                                if not argument_part:
                                    continue
                                yield send_sse_chunk({
                                    "id": msg_state_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": index,
                                                "function": {"arguments": argument_part},
                                            }]
                                        },
                                        "finish_reason": None,
                                    }]
                                })

                        if include_tool_extensions and execution["output"]:
                            previous_output = emitted_tool_result_output.get(tool_id)
                            if previous_output != execution["output"]:
                                emitted_tool_result_output[tool_id] = execution["output"]
                                yield send_sse_chunk(
                                    create_tool_result_extension_chunk(
                                        msg_state_id, model, execution
                                    )
                                )

                    def emit_content(raw_text):
                        if raw_text and msg_state_id:
                            yield send_content_delta(raw_text, msg_state_id)

                    def emit_stream_end(dify_chunk):
                        nonlocal stream_done
                        if stream_done:
                            return

                        if CONVERSATION_MEMORY_MODE == 2:
                            conversation_id = dify_chunk.get("conversation_id")
                            history = dify_chunk.get("conversation_history", [])
                            has_conversation_id = False
                            if history:
                                for msg in history:
                                    if msg.get("role") == "assistant":
                                        content = msg.get("content", "")
                                        if decode_conversation_id(content) is not None:
                                            has_conversation_id = True
                                            break
                            if conversation_id and not has_conversation_id and msg_state_id:
                                logger.info(f"[Debug] Inserting conversation_id in stream: {conversation_id}")
                                encoded = encode_conversation_id(conversation_id)
                                yield send_content_delta(encoded, msg_state_id)

                        final_chunk = {
                            "id": msg_state_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        stream_done = True

                    def process_tool_events(dify_chunk):
                        nonlocal msg_state_id
                        message_id = dify_chunk.get("message_id", "") or dify_chunk.get("id", "")
                        if not msg_state_id and message_id:
                            msg_state_id = message_id

                        for execution in extract_tool_executions_from_dify_event(dify_chunk):
                            merged, changed = update_tool_execution_snapshot(
                                tool_execution_snapshots, execution
                            )
                            if changed:
                                for event in emit_tool_execution_events(merged):
                                    yield event

                    def process_dify_chunk(dify_chunk):
                        nonlocal msg_state_id, stream_done
                        if stream_done:
                            return

                        event = dify_chunk.get("event")

                        if event == "agent_message":
                            message_id = dify_chunk.get("message_id", "") or dify_chunk.get("id", "")
                            if not msg_state_id and message_id:
                                msg_state_id = message_id
                            return

                        if should_forward_dify_content_chunk(dify_chunk):
                            current_answer = dify_chunk.get("answer", "")
                            if current_answer is None:
                                return

                            message_id = dify_chunk.get("message_id", "") or dify_chunk.get("id", "")
                            if not msg_state_id and message_id:
                                msg_state_id = message_id

                            yield from emit_content(current_answer)
                            return

                        if event == "agent_thought":
                            thought_id = dify_chunk.get("id", "")
                            tool = dify_chunk.get("tool", "")
                            logger.info(f"[Agent Thought] ID: {thought_id}, Tool: {tool}")
                            yield from process_tool_events(dify_chunk)
                            return

                        if event in {"agent_log", "node_finished"}:
                            yield from process_tool_events(dify_chunk)
                            return

                        if event == "message_file":
                            file_id = dify_chunk.get("id", "")
                            file_type = dify_chunk.get("type", "")
                            file_url = dify_chunk.get("url", "")
                            logger.info(f"[Message File] ID: {file_id}, Type: {file_type}, URL: {file_url}")
                            return

                        if is_dify_stream_terminal_event(dify_chunk):
                            yield from emit_stream_end(dify_chunk)

                    try:
                        async with client.stream(
                            'POST',
                            dify_endpoint,
                            json=dify_request,
                            headers={
                                **headers,
                                'Accept': 'text/event-stream',
                                'Cache-Control': 'no-cache',
                                'Connection': 'keep-alive'
                            }
                        ) as response:
                            line_reader = SseLineReader()

                            async for raw_bytes in response.aiter_raw():
                                if stream_done or not raw_bytes:
                                    break

                                try:
                                    for line in line_reader.feed(raw_bytes):
                                        if stream_done:
                                            break
                                        line = line.strip()
                                        if not line or not line.startswith('data: '):
                                            continue

                                        try:
                                            dify_chunk = json.loads(line[6:])
                                            log_raw_dify_event(dify_chunk)
                                            for item in process_dify_chunk(dify_chunk):
                                                yield item
                                        except json.JSONDecodeError as e:
                                            logger.error(f"JSON decode error: {str(e)}")
                                            continue
                                except Exception as e:
                                    logger.error(f"Error processing chunk: {str(e)}")
                                    continue

                            if not stream_done:
                                for line in line_reader.flush():
                                    if stream_done:
                                        break
                                    line = line.strip()
                                    if not line or not line.startswith('data: '):
                                        continue
                                    try:
                                        dify_chunk = json.loads(line[6:])
                                        log_raw_dify_event(dify_chunk)
                                        for item in process_dify_chunk(dify_chunk):
                                            yield item
                                    except json.JSONDecodeError as e:
                                        logger.error(f"JSON decode error on flush: {str(e)}")

                            if not stream_done:
                                for item in emit_stream_end({}):
                                    yield item
                    except Exception as e:
                        logger.error(f"Stream error: {str(e)}")

            return StreamingResponse(
                generate(),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache, no-transform',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    collected = await collect_dify_response_via_stream(
                        client,
                        dify_endpoint,
                        dify_request,
                        headers,
                    )
                    logger.info(
                        f"Collected via stream: answer_len={len(collected['answer'])}, "
                        f"tool_executions={len(collected['tool_executions'])}"
                    )
                    openai_response = transform_dify_to_openai(
                        {
                            "message_id": collected["message_id"],
                            "conversation_id": collected["conversation_id"],
                            "created": int(time.time()),
                        },
                        model=model,
                        include_tool_extensions=include_tool_extensions,
                        tool_executions=collected["tool_executions"],
                        answer_override=collected["answer"],
                    )
                    conversation_id = collected["conversation_id"]
                    
                    if conversation_id:
                        return JSONResponse(
                            content=openai_response,
                            headers={
                                'Conversation-Id': conversation_id
                            }
                        )
                    else:
                        return JSONResponse(content=openai_response)
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code if e.response is not None else 502
                error_msg = str(e)
                logger.error(f"Request failed: {error_msg}")
                return JSONResponse(status_code=status_code, content={
                    "error": {
                        "message": error_msg,
                        "type": "api_error",
                        "code": status_code
                    }
                })
            except httpx.RequestError as e:
                error_msg = f"Failed to connect to Dify: {repr(e)}"
                logger.error(error_msg)
                return JSONResponse(status_code=503, content={
                    "error": {
                        "message": error_msg,
                        "type": "api_error",
                        "code": "connection_error"
                    }
                })

    except Exception as e:
        logger.exception("Unexpected error occurred")
        return JSONResponse(status_code=500, content={
            "error": {
                "message": str(e),
                "type": "internal_error",
            }
        })

@router.get('/v1/models')
async def list_models():
    """返回可用的模型列表"""
    logger.info("Listing available models")
    
    await model_manager.refresh_model_info()
    available_models = model_manager.get_available_models()
    
    response = {
        "object": "list",
        "data": available_models
    }
    logger.info(f"Available models: {json.dumps(response, ensure_ascii=False)}")
    return JSONResponse(content=response)
