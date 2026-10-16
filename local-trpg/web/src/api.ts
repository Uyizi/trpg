export class ApiError extends Error {
  constructor(message: string, public readonly status: number) { super(message); }
}

export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: 'same-origin',
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = typeof payload?.detail === 'string' ? payload.detail : null;
    throw new ApiError(detail || ({ 409: '进度已经变化，请检查最新记录后再行动。', 422: '请检查输入内容。', 503: '本地主持人暂时不可用，请稍后重试。' }[response.status] ?? '操作未完成，请稍后重试。'), response.status);
  }
  return response.json() as Promise<T>;
}

export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : '暂时无法连接本地服务。请确认应用仍在运行，再重试。';
}

export async function downloadFile(path: string, filename: string) {
  const response = await fetch(path, { credentials: 'same-origin', signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new ApiError('下载未完成，请检查本地服务后重试。', response.status);
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
