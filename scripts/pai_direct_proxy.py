"""Loopback-only CONNECT relay for this PAI instance; preserves end-to-end TLS.

Pinned addresses were resolved with Alibaba DNS. Their actual local TUN route
must be checked as direct before transferring large files through this relay.
This does not proxy any cloud-side model or dataset downloads.
"""
import select
import socket
import socketserver

TARGETS = {
    'pai-dsw.cn-hangzhou.aliyuncs.com': ['101.37.132.1', '47.111.202.72'],
    'dsw-gateway-cn-hangzhou.data.aliyuncs.com': ['115.29.230.107', '101.37.69.95'],
}


class Relay(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(15)
        line = self.rfile.readline(4096).decode('ascii', errors='replace').strip()
        parts = line.split()
        if len(parts) != 3 or parts[0] != 'CONNECT':
            self.wfile.write(b'HTTP/1.1 405 Method Not Allowed\r\n\r\n')
            return
        host, _, port = parts[1].rpartition(':')
        if host not in TARGETS or port != '443':
            self.wfile.write(b'HTTP/1.1 403 Forbidden\r\n\r\n')
            return
        for _ in range(50):
            if self.rfile.readline(4096) in (b'\r\n', b'\n', b''):
                break
        else:
            return
        upstream = None
        for address in TARGETS[host]:
            try:
                upstream = socket.create_connection((address, 443), timeout=8)
                break
            except OSError:
                continue
        if upstream is None:
            self.wfile.write(b'HTTP/1.1 502 Bad Gateway\r\n\r\n')
            return
        with upstream:
            self.wfile.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            self.wfile.flush()
            self.connection.settimeout(30)
            upstream.settimeout(30)
            while True:
                readers, _, _ = select.select([self.connection, upstream], [], [], 120)
                if not readers:
                    return
                for source in readers:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.connection else self.connection).sendall(data)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    with Server(('127.0.0.1', 18789), Relay) as server:
        print('PAI direct relay listening on 127.0.0.1:18789', flush=True)
        server.serve_forever()
