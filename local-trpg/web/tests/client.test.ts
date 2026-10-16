import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { actionBlocked, GameClient } from '../src/client';
import type { PlayerView, TurnStatus } from '../src/types';

const view: PlayerView = {
  campaign_id: 'campaign-1', revision: 0, module_title: '备用冷却',
  player: { id: 'player', name: '林', location_id: 'hab', location: '生活舱', health: 'healthy', inventory: [] },
  tick: 0, phase: 'active', clocks: [{ label: '主回路热负荷', value: 0, limit: 12 }],
  locations: [{ id: 'hab', name: '生活舱' }], npcs: [], evidence: [], suggestions: [],
  transcript: [], outcome: null, active_turn_id: null, mock: false,
};
const status = (name: TurnStatus['status']): TurnStatus => ({
  turn_id: 'turn-1', status: name, stage: 'finished', is_terminal: !['running', 'queued'].includes(name),
  public_reply: name === 'needs_input' ? '你想对谁说？' : '', revision: 0, result: null, cancel_requested: false,
});
let posted: Array<Record<string, unknown>>;
let failSubmission: boolean;
let turnStatus: TurnStatus;
let turnReads: number;
let cancelPending: boolean;
let rejectionCode: number;
let serverView: PlayerView;
let turnNotFound: boolean;
let otherTurnStatus: TurnStatus;
const reply = (data: unknown, code = 200) => new Response(JSON.stringify(data), { status: code, headers: { 'Content-Type': 'application/json' } });

beforeEach(() => {
  localStorage.clear(); posted = []; failSubmission = false; turnStatus = status('committed'); turnReads = 0;
  cancelPending = false; rejectionCode = 0; serverView = view; turnNotFound = false;
  otherTurnStatus = { ...status('running'), turn_id: 'turn-2' };
  vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const path = String(input);
    if (path === '/api/health') return reply({ status: 'ok', backend: 'local', model_available: true, model_busy: false, mock: false });
    if (path === '/api/modules') return reply([{ module_id: 'reserve_cooling', title: '备用冷却', description: '调查事故', opening: '飞船失去动力。' }]);
    if (path === '/api/campaigns') return reply([]);
    if (path === '/api/campaigns/campaign-1') return reply(serverView);
    if (path.endsWith('/turns') && init?.method === 'POST') {
      posted.push(JSON.parse(String(init.body)));
      if (failSubmission) throw new TypeError('connection lost');
      if (rejectionCode) return reply({ detail: '进度已经变化' }, rejectionCode);
      return reply({ turn_id: 'turn-1', status: 'queued' }, 202);
    }
    if (path.endsWith('/turns/turn-1/cancel')) { turnStatus = cancelPending ? { ...status('running'), cancel_requested: true } : status('cancelled'); return reply(turnStatus); }
    if (path.endsWith('/turns/turn-1/resume')) { turnStatus = status('running'); return reply(turnStatus); }
    if (path.endsWith('/turns/turn-1')) { turnReads++; return turnNotFound ? reply({ detail: '没有这个回合' }, 404) : reply(turnStatus); }
    if (path.endsWith('/turns/turn-2')) return reply(otherTurnStatus);
    throw new Error(`Unexpected request: ${path}`);
  }));
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
const fields = { mode: 'in_character' as const, text: '检查阀组', speech_channel: 'local' as const, recipient_ids: [], private_aside: '先不说出猜测' };

// Removing persisted exact-body retry would consume a second action after an ambiguous network failure.
describe('GameClient action lifecycle', () => {
  it('retains the exact request body and ID after a lost receipt, including after reload; never auto-resends', async () => {
    const client = new GameClient();
    await client.openCampaign('campaign-1');
    failSubmission = true;
    expect(await client.submit(fields)).toBe(false);
    expect(client.getSnapshot().pending?.command.text).toBe('检查阀组');
    expect(posted).toHaveLength(1);
    client.dispose();
    const reloaded = new GameClient();
    await reloaded.start();
    expect(posted).toHaveLength(1);
    failSubmission = false;
    expect(await reloaded.retrySubmission()).toBe(true);
    expect(posted).toHaveLength(2);
    expect(posted[1]).toEqual(posted[0]);
    expect(posted[1].client_request_id).toBeTruthy();
    expect(reloaded.getSnapshot().pending).toBeNull();
    reloaded.dispose();
  });

  it.each(['committed', 'completed_ooc', 'needs_input', 'failed', 'cancelled', 'recoverable'] as const)('stops polling for terminal %s', async (name) => {
    vi.useFakeTimers(); turnStatus = status(name);
    const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
    await vi.advanceTimersByTimeAsync(15000);
    expect(client.getSnapshot().turn?.status).toBe(name);
    expect(turnReads).toBe(1);
    client.dispose();
  });

  it('links a clarification answer with a new request ID without advancing expected revision', async () => {
    vi.useFakeTimers(); turnStatus = status('needs_input');
    const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
    await vi.advanceTimersByTimeAsync(0);
    await client.submit({ ...fields, text: '我问机械师。' });
    expect(posted[1].reply_to_turn_id).toBe('turn-1');
    expect(posted[1].client_request_id).not.toBe(posted[0].client_request_id);
    expect(posted[1].expected_revision).toBe(0);
    client.dispose();
  });

  it('blocks new actions while recoverable and resumes only the original turn', async () => {
    vi.useFakeTimers(); turnStatus = status('recoverable');
    const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
    await vi.advanceTimersByTimeAsync(0);
    expect(await client.submit({ ...fields, text: '另一个行动' })).toBe(false);
    expect(posted).toHaveLength(1);
    await client.resumeTurn();
    expect(client.getSnapshot().turn?.status).toBe('running');
    expect(posted).toHaveLength(1);
    client.dispose();
  });

  it('treats cancellation as pending until the server confirms terminal status', async () => {
    vi.useFakeTimers(); turnStatus = status('running');
    const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
    await vi.advanceTimersByTimeAsync(0);
    cancelPending = true;
    await client.cancelTurn();
    expect(client.getSnapshot().turn?.cancel_requested).toBe(true);
    expect(await client.submit(fields)).toBe(false);
    turnStatus = status('cancelled');
    await vi.advanceTimersByTimeAsync(1100);
    expect(client.getSnapshot().turn?.status).toBe('cancelled');
    expect(await client.submit(fields)).toBe(true);
    client.dispose();
  });
});

// Historical results must not discard later progress or freshly projected command metadata.
it.each([1, 3])('keeps the authoritative revision 3 view when restoring a revision %i result', async (revision) => {
  vi.useFakeTimers();
  turnStatus = { ...status('committed'), revision: 3, result: { narration: '已完成', dialogue: [], checks: [], degraded: false, view: { ...view, revision, tick: revision } } };
  const client = new GameClient();
  await client.openCampaign('campaign-1'); await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0); client.dispose();
  serverView = {
    ...view, revision: 3, tick: 3, suggestions: ['通过工作频道联系'],
    suggestion_commands: { '通过工作频道联系': { mode: 'in_character', text: '联系工程师', speech_channel: 'broadcast', recipient_ids: [] } },
  };
  const restored = new GameClient();
  await restored.start(); await vi.advanceTimersByTimeAsync(0);
  expect(restored.getSnapshot().view?.revision).toBe(3);
  expect(restored.getSnapshot().view?.tick).toBe(3);
  expect(restored.getSnapshot().view?.suggestion_commands?.['通过工作频道联系'].speech_channel).toBe('broadcast');
  expect(restored.getSnapshot().view?.active_turn_id).toBeNull();
  expect(posted).toHaveLength(1);
  expect(await restored.submit({ ...fields, text: '再次检查' })).toBe(true);
  expect(posted[1].expected_revision).toBe(3);
  restored.dispose();
});

it('keeps another active turn blocked when an exact-body retry recovers an older receipt', async () => {
  vi.useFakeTimers();
  const client = new GameClient(); await client.openCampaign('campaign-1');
  failSubmission = true;
  await client.submit(fields); client.dispose();
  serverView = { ...view, revision: 3, tick: 3, active_turn_id: 'turn-2' };
  turnStatus = { ...status('committed'), revision: 3, result: { narration: '旧回合已完成', dialogue: [], checks: [], degraded: false, view: { ...view, revision: 1, tick: 1 } } };
  const restored = new GameClient();
  await restored.start(); await vi.advanceTimersByTimeAsync(0);
  failSubmission = false;
  expect(await restored.retrySubmission()).toBe(true);
  await vi.advanceTimersByTimeAsync(0);
  expect(posted[1]).toEqual(posted[0]);
  expect(restored.getSnapshot().view?.revision).toBe(3);
  expect(restored.getSnapshot().view?.active_turn_id).toBe('turn-2');
  expect(restored.getSnapshot().turnId).toBe('turn-2');
  expect(restored.getSnapshot().turn?.turn_id).toBe('turn-2');
  expect(actionBlocked(restored.getSnapshot())).toBe(true);
  expect(await restored.submit(fields)).toBe(false);
  otherTurnStatus = { ...status('committed'), turn_id: 'turn-2', revision: 4, result: { narration: '当前回合已完成', dialogue: [], checks: [], degraded: false, view: { ...view, revision: 4, tick: 4 } } };
  await vi.advanceTimersByTimeAsync(1100);
  expect(restored.getSnapshot().view?.revision).toBe(4);
  expect(restored.getSnapshot().view?.active_turn_id).toBeNull();
  expect(actionBlocked(restored.getSnapshot())).toBe(false);
  restored.dispose();
});

// A completed revision must replace stale state before the next action can be accepted.
it('uses the committed view revision for the next action and restores an active turn without submitting', async () => {
  vi.useFakeTimers();
  const client = new GameClient();
  await client.openCampaign('campaign-1');
  await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0);
  client.dispose();
  serverView = { ...view, revision: 3, tick: 3, active_turn_id: 'turn-1' };
  turnStatus = { ...status('committed'), revision: 4, result: { narration: '已完成', dialogue: [], checks: [], degraded: false, view: { ...view, revision: 4, tick: 4 } } };
  const restored = new GameClient();
  await restored.start();
  await vi.advanceTimersByTimeAsync(0);
  expect(posted).toHaveLength(1);
  expect(restored.getSnapshot().view?.tick).toBe(4);
  expect(restored.getSnapshot().view?.active_turn_id).toBeNull();
  await restored.submit({ ...fields, text: '再次检查' });
  expect(posted[1].expected_revision).toBe(4);
  restored.dispose();
});

it('refreshes a revision conflict without retrying or retaining a rejected command', async () => {
  const client = new GameClient();
  await client.openCampaign('campaign-1');
  rejectionCode = 409; serverView = { ...view, revision: 2, tick: 2 };
  expect(await client.submit(fields)).toBe(false);
  expect(client.getSnapshot().pending).toBeNull();
  expect(client.getSnapshot().view?.revision).toBe(2);
  expect(posted).toHaveLength(1);
  client.dispose();
});

it('retries an ambiguous server error with the same request identity', async () => {
  const client = new GameClient();
  await client.openCampaign('campaign-1');
  rejectionCode = 503;
  expect(await client.submit(fields)).toBe(false);
  rejectionCode = 0;
  expect(await client.retrySubmission()).toBe(true);
  expect(posted[1]).toEqual(posted[0]);
  client.dispose();
});

it('reconciles a missing persisted turn with the authoritative campaign instead of locking input forever', async () => {
  vi.useFakeTimers();
  const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0); client.dispose();
  turnNotFound = true;
  const restored = new GameClient(); await restored.start();
  await vi.advanceTimersByTimeAsync(0);
  expect(restored.getSnapshot().turnId).toBeNull();
  expect(actionBlocked(restored.getSnapshot())).toBe(false);
  expect(posted).toHaveLength(1);
  restored.dispose();
});

const liveStatus = (epoch = 1): TurnStatus => ({ ...status('running'), campaign_id: 'campaign-1', epoch, base_revision: 0,
  provisional: { campaign_id: 'campaign-1', turn_id: 'turn-1', epoch, base_revision: 0,
    dialogue: [{ event_id: 'reply-1', speaker: '顾成', speaker_id: 'mechanic', text: 'TEMPORARY_REPLY' }] } });

it('keeps temporary replies in memory only and clears them on poll error and cancel', async () => {
  vi.useFakeTimers(); turnStatus = liveStatus();
  const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0);
  expect(client.getSnapshot().turn?.provisional?.dialogue[0].text).toBe('TEMPORARY_REPLY');
  expect(localStorage.getItem('local-trpg.session.v1')).not.toContain('TEMPORARY_REPLY');
  vi.mocked(fetch).mockRejectedValueOnce(new TypeError('network lost'));
  await vi.advanceTimersByTimeAsync(1100);
  expect(client.getSnapshot().turn?.provisional).toBeNull();
  await vi.advanceTimersByTimeAsync(1100);
  expect(client.getSnapshot().turn?.provisional?.dialogue).toHaveLength(1);
  await client.cancelTurn();
  expect(client.getSnapshot().turn?.provisional).toBeNull();
  client.dispose();
});

it.each(['campaign', 'turn', 'epoch', 'revision'] as const)('drops temporary replies with mismatched %s identity', async mismatch => {
  vi.useFakeTimers(); turnStatus = liveStatus();
  if (mismatch === 'campaign') turnStatus.provisional!.campaign_id = 'another';
  if (mismatch === 'turn') turnStatus.provisional!.turn_id = 'another';
  if (mismatch === 'epoch') turnStatus.provisional!.epoch = 0;
  if (mismatch === 'revision') turnStatus.provisional!.base_revision = 3;
  const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0);
  expect(client.getSnapshot().turn?.provisional).toBeNull();
  client.dispose();
});

it('clears replies on campaign selection and ignores an older in-flight poll', async () => {
  vi.useFakeTimers(); turnStatus = liveStatus();
  const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0);
  let finishPoll!: (response: Response) => void;
  vi.mocked(fetch).mockImplementationOnce(() => new Promise(resolve => { finishPoll = resolve; }));
  await vi.advanceTimersByTimeAsync(1100);
  let finishSelection!: (response: Response) => void;
  vi.mocked(fetch).mockImplementationOnce(() => new Promise(resolve => { finishSelection = resolve; }));
  const selected = client.openCampaign('campaign-2');
  expect(client.getSnapshot().turn?.provisional ?? null).toBeNull();
  finishSelection(reply({ ...view, campaign_id: 'campaign-2' })); await selected;
  finishPoll(reply(liveStatus())); await vi.advanceTimersByTimeAsync(0);
  expect(client.getSnapshot().view?.campaign_id).toBe('campaign-2');
  expect(client.getSnapshot().turn).toBeNull();
  client.dispose();
});

it('ignores a lower execution epoch rather than resurrecting recovered speech', async () => {
  vi.useFakeTimers(); turnStatus = { ...liveStatus(3), provisional: null };
  const client = new GameClient(); await client.openCampaign('campaign-1'); await client.submit(fields);
  await vi.advanceTimersByTimeAsync(0);
  turnStatus = liveStatus(1); await vi.advanceTimersByTimeAsync(1100);
  expect(client.getSnapshot().turn?.epoch).toBe(3);
  expect(client.getSnapshot().turn?.provisional).toBeNull();
  client.dispose();
});

it('does not attach a late submit receipt to a newly selected campaign', async () => {
  const client = new GameClient(); await client.openCampaign('campaign-1');
  let finishSubmit!: (response: Response) => void;
  vi.mocked(fetch).mockImplementationOnce(() => new Promise(resolve => { finishSubmit = resolve; }));
  const submission = client.submit(fields);
  vi.mocked(fetch).mockResolvedValueOnce(reply({ ...view, campaign_id: 'campaign-2' }));
  await client.openCampaign('campaign-2');
  finishSubmit(reply({ turn_id: 'turn-1', status: 'queued' }, 202)); await submission;
  expect(client.getSnapshot().view?.campaign_id).toBe('campaign-2');
  expect(client.getSnapshot().turnId).toBeNull();
  expect(client.getSnapshot().turn).toBeNull();
  client.dispose();
});
