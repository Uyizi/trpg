import { afterEach, expect, it, vi } from 'vitest';
import { GameClient } from '../src/client';
const response = (data: unknown) => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); localStorage.clear(); });

it('polls an already loading model to ready without requesting another startup', async () => {
  vi.useFakeTimers(); localStorage.clear();
  let ready = false;
  let starts = 0;
  vi.stubGlobal('fetch', async (url: string) => {
    if (url === '/api/model/warmup') { starts++; return response({ status: 'loading' }); }
    if (url === '/api/health') return response({ status: ready ? 'ready' : 'loading', backend: 'runner', model_available: ready, model_busy: false, mock: false });
    return response([]);
  });
  const client = new GameClient();
  await client.start();
  ready = true;
  await vi.advanceTimersByTimeAsync(3000);
  expect(client.getSnapshot().health?.model_available).toBe(true);
  expect(starts).toBe(0);
  client.dispose();
});

it('starts only on explicit request and then refreshes actual health', async () => {
  localStorage.clear(); let starts = 0;
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    if (url === '/api/model/warmup' && init?.method === 'POST') { starts++; return response({ status: 'loading' }); }
    if (url === '/api/health') return response({ status: starts ? 'ready' : 'unavailable', backend: 'runner', model_available: starts > 0, model_busy: false, mock: false });
    return response([]);
  });
  const client = new GameClient();
  await client.start();
  expect(starts).toBe(0);
  expect(await client.warmupModel()).toBe(true);
  expect(starts).toBe(1);
  expect(client.getSnapshot().health?.model_available).toBe(true);
  client.dispose();
});
