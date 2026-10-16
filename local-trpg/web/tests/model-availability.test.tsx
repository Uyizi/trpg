import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ModelAvailability } from '../src/ModelAvailability';
import type { Health } from '../src/types';
const health: Health = { status: 'unavailable', backend: 'runner', model_available: false, model_busy: false, mock: false };
afterEach(cleanup);
it('offers a manual start only for an unavailable model and does not trigger it on render', () => {
  const start = vi.fn();
  render(<ModelAvailability health={health} warming={false} onWarmup={start} />);
  expect(start).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: '启动主持人' }));
  expect(start).toHaveBeenCalledOnce();
});
it('distinguishes loading from unavailable and prevents a second warmup', () => {
  render(<ModelAvailability health={{ ...health, status: 'loading' }} warming={false} onWarmup={vi.fn()} />);
  expect(screen.getByText('主持人准备中')).toBeTruthy();
  expect(screen.queryByRole('button', { name: '启动主持人' })).toBeNull();
});
it('removes the start action when the real model is ready', () => {
  render(<ModelAvailability health={{ ...health, status: 'ready', model_available: true }} warming={false} onWarmup={vi.fn()} />);
  expect(screen.getByText('本地主持人在线')).toBeTruthy();
  expect(screen.queryByRole('button')).toBeNull();
});
