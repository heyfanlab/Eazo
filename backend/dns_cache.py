# 模型服务域名的 IPv4 解析：带超时、备用解析和缓存。
#
# 为什么需要：线上函数在 Vercel 新加坡，百炼域名 dashscope.aliyuncs.com 会 CNAME 到阿里云
# 国内的智能 DNS（gtm-cn-…）。从 AWS 新加坡解析它的 A 记录偶尔会卡几十秒再失败——
# 实测一次识图卡 61s 报 Errno 99、一次生成建议卡 41s 报 Errno -9（都是 A 记录没解析出来，
# 只剩 IPv6 或无地址可用）。getaddrinfo 不受 socket 超时约束，原有的建连超时管不到它。
#
# 做法（只解析 IPv4，Vercel 函数没有 IPv6 出口）：
#   1. 缓存未过期 → 直接用；
#   2. 系统 DNS，最多等 SYSTEM_DNS_TIMEOUT_S；
#   3. DNS-over-HTTPS：1.1.1.1 / 8.8.8.8，用 IP 直连，不依赖系统 DNS；
#   4. 都失败 → 用上一次成功的结果（即使已过期）；从没成功过才报错。
# TCP 连到解析出的 IP，TLS 握手的 SNI 和证书校验仍按原域名进行（httpcore 用请求的
# origin.host 做 server_hostname），所以安全性不变。
import ipaddress
import logging
import socket
import threading
import time

import httpcore
import httpx

logger = logging.getLogger(__name__)

SYSTEM_DNS_TIMEOUT_S = 2.0
DOH_TIMEOUT_S = 3.0
CACHE_TTL_S = 600
_DOH_ENDPOINTS = (
    ("https://1.1.1.1/dns-query", {"accept": "application/dns-json"}),
    ("https://8.8.8.8/resolve", {}),
)

_cache: dict[str, tuple[list[str], float]] = {}
_cache_lock = threading.Lock()


def _system_ipv4(host: str) -> list[str]:
    """在守护线程里跑 getaddrinfo，超时就放弃（线程自行结束，结果丢弃）。"""
    box: dict = {}

    def work():
        try:
            infos = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
            box["ips"] = list(dict.fromkeys(i[4][0] for i in infos))
        except Exception as e:  # noqa: BLE001 —— 任何解析失败都交给下一级
            box["err"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(SYSTEM_DNS_TIMEOUT_S)
    if "ips" in box and box["ips"]:
        return box["ips"]
    raise OSError(f"system DNS: {box.get('err') or f'timeout {SYSTEM_DNS_TIMEOUT_S}s'}")


def _doh_ipv4(host: str) -> list[str]:
    errors = []
    for url, headers in _DOH_ENDPOINTS:
        try:
            r = httpx.get(url, params={"name": host, "type": "A"}, headers=headers,
                          timeout=DOH_TIMEOUT_S)
            r.raise_for_status()
            ips = [a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1]
            if ips:
                return ips
            errors.append(f"{url}: no A record")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {type(e).__name__}")
    raise OSError("DoH: " + "; ".join(errors))


def resolve_ipv4(host: str) -> list[str]:
    try:
        ipaddress.IPv4Address(host)
        return [host]
    except ValueError:
        pass
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(host)
    if cached and now - cached[1] < CACHE_TTL_S:
        return cached[0]

    start = time.perf_counter()
    errors = []
    for name, fn in (("system", _system_ipv4), ("doh", _doh_ipv4)):
        try:
            ips = fn(host)
        except OSError as e:
            errors.append(str(e))
            continue
        with _cache_lock:
            _cache[host] = (ips, time.monotonic())
        if name != "system" or time.perf_counter() - start > 1:
            logger.warning("dns host=%s 经 %s 解析 耗时=%.1fs ips=%s 前序失败=%s",
                           host, name, time.perf_counter() - start, ips, errors)
        return ips
    if cached:
        logger.warning("dns host=%s 解析全部失败，沿用上次结果 ips=%s 失败=%s", host, cached[0], errors)
        return cached[0]
    raise httpcore.ConnectError(f"cannot resolve {host}: {'; '.join(errors)}")


def prewarm(host: str) -> None:
    """冷启动时后台先解析一次，第一条请求就能命中缓存。失败无所谓，请求时会再试。"""
    def work():
        try:
            resolve_ipv4(host)
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=work, daemon=True).start()


class CachedDNSBackend(httpcore.NetworkBackend):
    """httpcore 网络后端：TCP 连到 resolve_ipv4 给出的 IP，逐个尝试。"""

    def __init__(self):
        self._inner = httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        last_exc = None
        for ip in resolve_ipv4(host):
            try:
                return self._inner.connect_tcp(ip, port, timeout=timeout,
                                               local_address=local_address,
                                               socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                last_exc = e
        raise last_exc or httpcore.ConnectError(f"no address for {host}")

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    def sleep(self, seconds):
        self._inner.sleep(seconds)
