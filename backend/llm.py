# 模型调用封装（文档十二：所有模型调用经过本模块，带 1 次重试和 30s 超时）。
# chat：Agent 主循环用；embed：RAG 用；vision：食物照片识别用。
import base64
import logging
import threading
import time

import httpx
from openai import DefaultHttpxClient, OpenAI

from . import config

logger = logging.getLogger(__name__)

_client = None

# 线上函数在 Vercel 新加坡，百炼在北京，每次调用都跨境，偶发建连卡死
# （实测一次识图卡 61s 后报 ConnectError: [Errno 99]）。三处加固：
# 1) connect=5s：建连超时按「单个 IP」计。百炼有 4 个 IPv4，原先跟随总超时 20/30s，
#    卡住的 IP 要等满才换下一个；现在 5 秒就换。read 超时保持原值，不影响正常生成。
# 2) local_address=0.0.0.0：只走 IPv4。Vercel 函数没有 IPv6 出口，试 IPv6 只会报 Errno 99。
# 3) retries=1：仅「建连失败」时由传输层自动重连一次。请求还没发出去，不会重复计费；
#    请求发出后的读超时不在此列，仍由 chat 的 1 次重试 / vision 的调用方兜底处理。
CHAT_TIMEOUT = httpx.Timeout(30, connect=5)
VISION_TIMEOUT = httpx.Timeout(20, connect=5)


_phase_sink = threading.local()


class _PhaseTracingClient(DefaultHttpxClient):
    """调用方在当前线程挂一个 list（_phase_sink.events）时，把 httpcore 各阶段
    （建连/TLS/上传请求体/等响应头）的时间点追加进去，用于定位跨境卡在哪一步。
    没挂 list 时什么都不做。list 由调用方持有，工作线程被总时长上限放弃后，
    调用方仍能读到它已经走到了哪一步。"""

    def send(self, request, **kwargs):
        events = getattr(_phase_sink, "events", None)
        if events is not None:
            t0 = time.perf_counter()

            def trace(name, info):
                if name.endswith((".started", ".complete", ".failed")):
                    events.append(f"{name.split('.', 1)[1]}@{time.perf_counter() - t0:.1f}s")
            request.extensions["trace"] = trace
        return super().send(request, **kwargs)


def _http_client() -> httpx.Client:
    return _PhaseTracingClient(transport=httpx.HTTPTransport(
        local_address="0.0.0.0",
        retries=1,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    ))


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=CHAT_TIMEOUT,
            max_retries=0,
            http_client=_http_client(),
        )
    return _client


def chat(messages: list[dict], tools: list[dict] | None = None):
    """调 chat 模型，返回 choices[0].message。失败重试 1 次，再失败抛异常（调用方兜底）。"""
    kwargs = {"model": config.CHAT_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    for attempt in range(2):
        try:
            start = time.perf_counter()
            resp = get_client().chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            called = [tc.function.name for tc in (msg.tool_calls or [])]
            logger.info(
                "llm_call model=%s tools=%s 耗时=%.1fs",
                config.CHAT_MODEL, called or "无(最终输出)", time.perf_counter() - start,
            )
            return msg
        except Exception:
            if attempt == 0:
                logger.warning("chat 调用失败，重试第1次", exc_info=True)
                time.sleep(1)
            else:
                logger.error("chat 调用重试后仍失败", exc_info=True)
                raise


def vision(image_bytes: bytes, prompt: str, mime: str = "image/jpeg",
           phases: list | None = None) -> str:
    """视觉模型识别图片，返回文本。失败抛异常（调用方兜底）。
    phases：传入一个 list 时，记录本次 HTTP 各阶段时间点（见 _PhaseTracingClient）。"""
    phases = [] if phases is None else phases
    _phase_sink.events = phases
    b64 = base64.b64encode(image_bytes).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    # 单次尝试 + 20s per-call 超时：视觉识别正常几秒内就该回；不重试，避免
    # 慢网络下「30s 超时 → 重试 → 再 30s」放大成 60s+，让用户干等（调用方有兜底）。
    try:
        start = time.perf_counter()
        resp = get_client().with_options(timeout=VISION_TIMEOUT).chat.completions.create(
            model=config.VISION_MODEL, messages=messages
        )
        logger.info(
            "llm_call model=%s vision 耗时=%.1fs 图片=%dKB 阶段=%s",
            config.VISION_MODEL, time.perf_counter() - start, len(image_bytes) // 1024, phases,
        )
        return resp.choices[0].message.content
    except Exception:
        logger.error("vision 调用失败（不重试，走调用方兜底）图片=%dKB 阶段=%s",
                     len(image_bytes) // 1024, phases, exc_info=True)
        raise
    finally:
        _phase_sink.events = None


def embed(texts: list[str]) -> list[list[float]]:
    """文本向量化。失败重试 1 次，再失败抛异常（调用方决定兜底）。"""
    for attempt in range(2):
        try:
            start = time.perf_counter()
            resp = get_client().embeddings.create(
                model=config.EMBED_MODEL, input=texts
            )
            logger.info(
                "llm_call model=%s 条数=%d 耗时=%.1fs",
                config.EMBED_MODEL, len(texts), time.perf_counter() - start,
            )
            # 按 index 排序，保证与输入顺序一致
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        except Exception:
            if attempt == 0:
                logger.warning("embed 调用失败，重试第1次", exc_info=True)
                time.sleep(1)
            else:
                logger.error("embed 调用重试后仍失败", exc_info=True)
                raise
