"""Windows physical-interface transport for WSL web retrieval.

Each socket is bound to a physical adapter with IP_UNICAST_IF. DNS uses
verified HTTPS to Alibaba DNS over that same adapter. No proxy fallback.
"""
import base64
from contextlib import contextmanager
import http.client
import ipaddress
import json
import socket
import ssl
import struct
import subprocess
import time
from urllib.parse import urlencode

_ADAPTER = None
_DNS = {}
DIAGNOSTICS = []


def remaining(deadline, stage):
    budget = deadline - time.monotonic()
    if budget <= 0:
        exc = TimeoutError('Physical transport deadline exceeded')
        exc.stage = stage
        raise exc
    return budget


@contextmanager
def timed(stage):
    started = time.monotonic()
    record = {'stage': stage, 'status': 'ok'}
    try:
        yield
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__)
        if not hasattr(exc, 'stage'):
            exc.stage = stage
        raise
    finally:
        record['seconds'] = round(time.monotonic() - started, 3)
        DIAGNOSTICS.append(record)


def interface(timeout=10):
    global _ADAPTER
    if _ADAPTER is not None:
        return _ADAPTER
    code = """
    $physical = @(Get-NetAdapter -Physical | Where-Object {$_.Status -eq 'Up'} | Select-Object -ExpandProperty ifIndex)
    Get-NetIPConfiguration | Where-Object {$_.InterfaceIndex -in $physical -and $_.IPv4DefaultGateway} |
      Sort-Object {$_.NetIPv4Interface.InterfaceMetric} | Select-Object -First 1 |
      ForEach-Object {[PSCustomObject]@{index=$_.InterfaceIndex;ip=$_.IPv4Address.IPAddress}} |
      ConvertTo-Json -Compress
    """
    with timed('physical_interface'):
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand',
             base64.b64encode(code.encode('utf-16le')).decode()],
            capture_output=True, text=True, timeout=min(10, timeout), check=True)
        value = json.loads(result.stdout)
        if not value or not value.get('index') or not isinstance(value.get('ip'), str):
            raise ConnectionError('No physical network adapter with IPv4 gateway')
        _ADAPTER = value
        return value


def connect(address, port, timeout):
    deadline = time.monotonic() + timeout
    adapter = interface(timeout=remaining(deadline, 'physical_interface'))
    budget = remaining(deadline, 'physical_connect')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with timed('physical_connect'):
            sock.settimeout(budget)
            # Winsock expects the interface index in network byte order.
            sock.setsockopt(socket.IPPROTO_IP, 31, struct.pack('!I', adapter['index']))
            sock.bind((adapter['ip'], 0))
            sock.connect((address, port))
            return sock
    except BaseException:
        sock.close()
        raise


def resolve(host, port, *, timeout=6):
    key = (host, port)
    if key in _DNS:
        return _DNS[key]
    deadline = time.monotonic() + min(6, timeout)
    with timed('physical_dns'):
        try:
            addresses = [str(ipaddress.IPv4Address(host))]
        except ipaddress.AddressValueError:
            conn = http.client.HTTPSConnection('dns.alidns.com', timeout=remaining(deadline, 'physical_dns'))
            raw = connect('223.5.5.5', 443, remaining(deadline, 'physical_dns'))
            try:
                raw.settimeout(remaining(deadline, 'physical_dns_tls'))
                transport = ssl.create_default_context().wrap_socket(raw, server_hostname='dns.alidns.com')
                conn.sock = transport
                transport.settimeout(remaining(deadline, 'physical_dns_headers'))
                conn.request('GET', '/resolve?' + urlencode({'name': host, 'type': 'A'}),
                             headers={'Accept': 'application/dns-json'})
                transport.settimeout(remaining(deadline, 'physical_dns_headers'))
                response = conn.getresponse()
                if response.status != 200:
                    raise ConnectionError('Direct DNS request failed')
                body = bytearray()
                while len(body) < 65536:
                    transport.settimeout(remaining(deadline, 'physical_dns_body'))
                    block = response.read1(min(8192, 65536 - len(body)))
                    if not block:
                        break
                    body.extend(block)
                data = json.loads(body)
                addresses = [r['data'] for r in data.get('Answer', []) if r.get('type') == 1]
                if data.get('Status') != 0 or not addresses:
                    raise ConnectionError('Direct DNS returned no IPv4 records')
            finally:
                conn.close()
                raw.close()
        rows = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port))
                for ip in addresses]
        _DNS[key] = rows
        return rows
