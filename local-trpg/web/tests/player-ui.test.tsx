import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useState, useSyncExternalStore } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { ActionComposer, Conversation, PlayerPanel } from '../src/Player';
import { GameClient } from '../src/client';
import type { CommandFields, PlayerView } from '../src/types';

const view: PlayerView = {
  campaign_id: 'c', revision: 1, module_title: '备用冷却',
  player: { id: 'player', name: '林', location_id: 'hab', location: '生活舱', health: 'healthy', inventory: [] },
  tick: 1, phase: 'active', clocks: [], locations: [],
  npcs: [{ id: 'mechanic', name: '顾成', role: '机械师', location_id: null, present: false }],
  evidence: [], suggestions: ['观察舱门'],
  transcript: [{ turn_id: 't', speaker: 'DM', text: '<img src=x onerror=alert(1)>', tick: 1 }],
  outcome: null, active_turn_id: null, mock: false,
};
afterEach(cleanup);

it('shows the d20 character sheet and current opponents without ship-only controls', () => {
  const adventure: PlayerView = { ...view, ruleset: 'pinebrook_d20', skill_names: { perception: '察觉' },
    player: { ...view.player, role: '半身人游荡者', hp: 7, max_hp: 11, ac: 14, skills: { perception: 3 },
      attacks: [{ id: 'shortsword', name: '短剑', bonus: 5, dice: 1, damage_bonus: 3 }],
      ability: { id: 'first_strike', name: '首轮突袭', uses: null } },
    combat: { round: 2, enemies: [{ id: 'icicle_1', name: '活冰锥1', condition: '受伤' }] } };
  render(<><PlayerPanel view={adventure} /><ActionComposer view={adventure} disabled={false} turn={null} pending={null} onSubmit={async () => true} /></>);
  expect(screen.getByText('7 / 11')).toBeTruthy();
  expect(screen.getByText('护甲 14')).toBeTruthy();
  expect(screen.getByText('察觉')).toBeTruthy();
  expect(screen.getByText('短剑')).toBeTruthy();
  expect(screen.getByText('活冰锥1')).toBeTruthy();
  expect(screen.queryByText('工作频道 · 广播')).toBeNull();
  expect(screen.queryByText(/使用2d6/)).toBeNull();
});

it('shows an adventure ending without fabricated ship or cell counts', () => {
  const adventure: PlayerView = { ...view, ruleset: 'pinebrook_d20',
    outcome: { kind: 'adventure', status: 'completed', summary: '护送完成，幼龙留在母亲身边。', wyrmling_returned: true, care_accepted: false },
    ending_summary: ['护送完成，幼龙留在母亲身边。'] };
  render(<Conversation view={adventure} turn={null} />);
  expect(screen.getByText('护送完成，幼龙留在母亲身边。')).toBeTruthy();
  expect(screen.queryByText('细胞保全')).toBeNull();
  expect(screen.queryByText('同路号')).toBeNull();
});

it('shows the players own character sheet without supplying an adventure solution', () => {
  render(<PlayerPanel view={{ ...view, player: { ...view.player, skills: { observe: 2, technical: 1 } } }} />);
  expect(screen.getByText('角色能力')).toBeTruthy();
  expect(screen.getByText('观察')).toBeTruthy();
  expect(screen.queryByText('离港前核对货物')).toBeNull();
});

it('keeps DM questions usable after the adventure ends and never submits them as actions', async () => {
  const submit = vi.fn(async () => true);
  const ended = { ...view, outcome: { crew_saved: 6, cells_saved: 0, ship_saved: true, cause_verified: false, captain_action_verified: false } };
  render(<ActionComposer view={ended} disabled={false} turn={null} pending={null} onSubmit={submit} />);
  fireEvent.click(screen.getByRole('button', { name: /向主持人提问/ }));
  expect(submit).not.toHaveBeenCalled();
  expect((screen.getByLabelText('你的行动') as HTMLTextAreaElement).value).toBe('');
  fireEvent.change(screen.getByLabelText('你的行动'), { target: { value: '这个结局是什么意思？' } });
  fireEvent.click(screen.getByRole('button', { name: '询问主持人' }));
  expect(submit).toHaveBeenCalledWith(expect.objectContaining({ mode: 'ooc', text: '这个结局是什么意思？', private_aside: '', recipient_ids: [] }));
});

it('keeps saved model table talk separate and does not falsely label old canned replies as Qwen', () => {
  render(<Conversation view={{ ...view, table_talk: [
    { turn_id: 'old', question: '旧问题', answer: '旧固定答复', reply_source: 'legacy', created: 1 },
    { turn_id: 'new', question: '我的追问', answer: '模型的实际回答', reply_source: 'model', created: 2 },
  ] }} turn={null} />);
  fireEvent.click(screen.getByRole('button', { name: '桌外对话' }));
  expect(screen.getByText('模型的实际回答')).toBeTruthy();
  expect(screen.getByText('Qwen · 主持人')).toBeTruthy();
  expect(screen.queryByText('旧固定答复')).toBeNull();
});

it('returns to new adventure results after a table conversation instead of following old Qwen answers', () => {
  const saved = { ...view, table_talk: [{ turn_id: 'talk', question: '规则问题', answer: '旧桌外回答', reply_source: 'model' as const, created: 2 }] };
  const { rerender } = render(<Conversation view={saved} turn={null} />);
  fireEvent.click(screen.getByRole('button', { name: '桌外对话' }));
  expect(screen.getByText('旧桌外回答')).toBeTruthy();
  rerender(<Conversation view={{ ...saved, transcript: [...saved.transcript, { turn_id: 'action', speaker: 'DM', text: '刚刚实际发生的结果', tick: 2 }] }} turn={{ turn_id: 'action', mode: 'in_character', status: 'committed', stage: '已完成', is_terminal: true, public_reply: '', revision: 2, result: null, cancel_requested: false }} />);
  expect(screen.getByText('刚刚实际发生的结果')).toBeTruthy();
  expect(screen.queryByText('旧桌外回答')).toBeNull();
});

it('switches the editor with the visible channel, and keeps a completed result visible until an explicit DM question', () => {
  const submit = vi.fn(async () => true);
  const ended = { ...view, outcome: { crew_saved: 6, cells_saved: 0, ship_saved: true, cause_verified: false, captain_action_verified: false } };
  function Paired() {
    const [mode, setMode] = useState<CommandFields['mode']>('in_character');
    return <><Conversation view={ended} turn={null} mode={mode} onModeChange={setMode} /><ActionComposer view={ended} mode={mode} onModeChange={setMode} disabled={false} turn={null} pending={null} onSubmit={submit} /></>;
  }
  render(<Paired />);
  expect(screen.getByText('航行告一段落')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '桌外对话' }));
  expect((screen.getByLabelText('输入方式') as HTMLSelectElement).value).toBe('ooc');
  fireEvent.change(screen.getByLabelText('你的行动'), { target: { value: '我想知道结果' } });
  fireEvent.click(screen.getByRole('button', { name: '询问主持人' }));
  expect(submit).toHaveBeenCalledWith(expect.objectContaining({ mode: 'ooc' }));
  fireEvent.click(screen.getByRole('button', { name: '冒险记录' }));
  expect(screen.getByText('航行告一段落')).toBeTruthy();
});

it('renders transcript content as text without inserting untrusted markup', () => {
  const { container } = render(<Conversation view={view} turn={null} />);
  expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy();
  expect(container.querySelector('img')).toBeNull();
});

it('uses actor IDs to distinguish same-name player, NPC, and DM entries', () => {
  const sameName: PlayerView = { ...view, player: { ...view.player, name: '主持人' },
    npcs: [{ id: 'doctor', name: '主持人', role: '医生', location_id: null, present: false }],
    transcript: [
      { turn_id: 't', speaker: '主持人', speaker_id: 'player', text: '玩家的问题', tick: 1 },
      { turn_id: 't', speaker: 'DM', speaker_id: 'dm', text: '主持人的叙述', tick: 1 },
      { turn_id: 't', speaker: '主持人', speaker_id: 'doctor', text: '医生的回答', tick: 1 },
    ],
  };
  const { container } = render(<Conversation view={sameName} turn={null} />);
  expect(container.querySelector('.player-entry')?.textContent).toContain('玩家的问题');
  expect(container.querySelector('.dm-entry')?.textContent).toContain('主持人的叙述');
  expect(container.querySelector('.npc-entry')?.textContent).toContain('医生的回答');
});

it('submits free text, private recipients and a separate DM aside without overwriting channel', async () => {
  const submit = vi.fn(async () => true);
  render(<ActionComposer view={view} disabled={false} turn={null} pending={null} onSubmit={submit} />);
  fireEvent.change(screen.getByLabelText('你的行动'), { target: { value: '问顾成工单在哪' } });
  fireEvent.change(screen.getByLabelText('发言范围'), { target: { value: 'private' } });
  fireEvent.click(screen.getByLabelText('顾成'));
  fireEvent.change(screen.getByLabelText('给 DM 的私下说明'), { target: { value: '我在试探他。' } });
  fireEvent.click(screen.getByRole('button', { name: '采取行动' }));
  expect(submit).toHaveBeenCalledWith({ mode: 'in_character', text: '问顾成工单在哪', speech_channel: 'private', recipient_ids: ['mechanic'], private_aside: '我在试探他。' });
});

it('keeps a failed submission draft so the player can recover it', async () => {
  render(<ActionComposer view={view} disabled={false} turn={null} pending={null} onSubmit={async () => false} />);
  const input = screen.getByLabelText('你的行动') as HTMLTextAreaElement;
  fireEvent.change(input, { target: { value: '检查阀组' } });
  fireEvent.click(screen.getByRole('button', { name: '采取行动' }));
  await Promise.resolve();
  expect(input.value).toBe('检查阀组');
});

it('lets suggestions fill the editor without making them the only action choice', () => {
  const submit = vi.fn();
  render(<ActionComposer view={view} disabled={false} turn={null} pending={null} onSubmit={submit} />);
  fireEvent.click(screen.getByRole('button', { name: '观察舱门' }));
  expect((screen.getByLabelText('你的行动') as HTMLTextAreaElement).value).toBe('观察舱门');
  expect(submit).not.toHaveBeenCalled();
});

it('selects the work channel when its suggestion is clicked, then submits broadcast', () => {
  const text = '使用工作频道询问当前情况';
  const submit = vi.fn(async () => true);
  const suggestedView: PlayerView = { ...view, suggestions: [text], suggestion_commands: {
    [text]: { text, mode: 'in_character', speech_channel: 'broadcast', recipient_ids: [] },
  } };
  render(<ActionComposer view={suggestedView} disabled={false} turn={null} pending={null} onSubmit={submit} />);
  fireEvent.change(screen.getByLabelText('发言范围'), { target: { value: 'private' } });
  fireEvent.click(screen.getByLabelText('顾成'));
  fireEvent.change(screen.getByLabelText('输入方式'), { target: { value: 'ooc' } });
  expect(screen.queryByRole('button', { name: text })).toBeNull();
  fireEvent.change(screen.getByLabelText('输入方式'), { target: { value: 'in_character' } });
  fireEvent.click(screen.getByRole('button', { name: text }));
  expect((screen.getByLabelText('输入方式') as HTMLSelectElement).value).toBe('in_character');
  expect((screen.getByLabelText('发言范围') as HTMLSelectElement).value).toBe('broadcast');
  expect(submit).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: '采取行动' }));
  expect(submit).toHaveBeenCalledWith({ text, mode: 'in_character', speech_channel: 'broadcast', recipient_ids: [], private_aside: '' });
});

it('clears the retained draft when an explicit retry is accepted, preventing accidental duplicate actions', () => {
  const pending = { command: { client_request_id: 'retry-1', expected_revision: 1, reply_to_turn_id: null, mode: 'in_character' as const, text: '检查阀组', speech_channel: 'local' as const, recipient_ids: [], private_aside: '私下观察' } };
  const { rerender } = render(<ActionComposer view={view} disabled={true} turn={null} pending={pending} acceptedRequestId={null} onSubmit={async () => true} />);
  expect((screen.getByLabelText('你的行动') as HTMLTextAreaElement).value).toBe('检查阀组');
  rerender(<ActionComposer view={view} disabled={false} turn={null} pending={null} acceptedRequestId="retry-1" onSubmit={async () => true} />);
  expect((screen.getByLabelText('你的行动') as HTMLTextAreaElement).value).toBe('');
  expect((screen.getByLabelText('给 DM 的私下说明') as HTMLTextAreaElement).value).toBe('');
});

it('shows validated replies before commit and replaces them with one canonical entry', () => {
  const activeView = { ...view, active_turn_id: 'live' };
  const preview = { campaign_id: 'c', turn_id: 'live', epoch: 1, base_revision: 1,
    dialogue: [{ event_id: 'speech-1', speaker: '顾成', speaker_id: 'mechanic', text: '<b>我收到消息了。</b>' }] };
  const running = { campaign_id: 'c', turn_id: 'live', epoch: 1, base_revision: 1, status: 'running' as const,
    stage: '相关船员回应', is_terminal: false, public_reply: '', revision: 1, result: null, cancel_requested: false, provisional: preview };
  const { rerender, container } = render(<Conversation view={activeView} turn={running} />);
  expect(screen.getByText('<b>我收到消息了。</b>')).toBeTruthy();
  expect(screen.getByText('本轮尚未完成，回应暂未写入记录。')).toBeTruthy();
  expect(container.querySelector('b')).toBeNull();
  rerender(<Conversation view={{ ...activeView, revision: 2, active_turn_id: null, transcript: [...view.transcript, {
    turn_id: 'live', speaker: '顾成', speaker_id: 'mechanic', text: '<b>我收到消息了。</b>', tick: 2,
  }] }} turn={{ ...running, status: 'committed', is_terminal: true }} />);
  expect(screen.getAllByText('<b>我收到消息了。</b>')).toHaveLength(1);
  expect(screen.queryByText('本轮尚未完成，回应暂未写入记录。')).toBeNull();
});

it.each(['cancelled', 'failed', 'recoverable'] as const)('removes temporary replies when %s', status => {
  const turn = { campaign_id: 'c', turn_id: 'live', epoch: 1, base_revision: 1, status, stage: '', is_terminal: true,
    public_reply: '', revision: 1, result: null, cancel_requested: false,
    provisional: { campaign_id: 'c', turn_id: 'live', epoch: 1, base_revision: 1,
      dialogue: [{ event_id: 'speech', speaker: '顾成', speaker_id: 'mechanic', text: 'Temporary speech' }] } };
  render(<Conversation view={{ ...view, active_turn_id: 'live' }} turn={turn} />);
  expect(screen.queryByText('Temporary speech')).toBeNull();
});

it('displays a polled reply during a running turn before the final response exists', async () => {
  vi.useFakeTimers(); localStorage.clear();
  const active = { ...view, active_turn_id: 'live' };
  const dialogue = [{ event_id: 'speech', speaker: '顾成', speaker_id: 'mechanic', text: '先收到的回答' }];
  let completed = false;
  vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
    const path = String(input);
    const running = { campaign_id: 'c', turn_id: 'live', epoch: 1, base_revision: 1, status: 'running', is_terminal: false,
      stage: '相关船员回应', public_reply: '', revision: 1, result: null, cancel_requested: false,
      provisional: { campaign_id: 'c', turn_id: 'live', epoch: 1, base_revision: 1, dialogue } };
    const data = path.endsWith('/turns/live') ? completed ? { ...running, status: 'committed', is_terminal: true, provisional: null,
      result: { narration: '已完成', dialogue, checks: [], degraded: false, view: { ...active, revision: 2, active_turn_id: null,
        transcript: [...view.transcript, { ...dialogue[0], turn_id: 'live', tick: 2 }] } } } : running : active;
    return new Response(JSON.stringify(data), { status: 200 });
  }));
  const client = new GameClient();
  function ConnectedConversation() {
    const state = useSyncExternalStore(client.subscribe, client.getSnapshot);
    return state.view ? <Conversation view={state.view} turn={state.turn} /> : null;
  }
  try {
    render(<ConnectedConversation />);
    await act(async () => { await client.openCampaign('c'); await vi.advanceTimersByTimeAsync(0); });
    expect(client.getSnapshot().turn?.result).toBeNull();
    expect(screen.getByText('先收到的回答')).toBeTruthy();
    expect(screen.getByText('本轮尚未完成，回应暂未写入记录。')).toBeTruthy();
    expect(localStorage.getItem('local-trpg.session.v1')).not.toContain('先收到的回答');
    completed = true;
    await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
    expect(client.getSnapshot().view?.revision).toBe(2);
    expect(screen.getAllByText('先收到的回答')).toHaveLength(1);
    expect(screen.queryByText('本轮尚未完成，回应暂未写入记录。')).toBeNull();
  } finally { client.dispose(); vi.useRealTimers(); vi.unstubAllGlobals(); }
});

it('offers current adventure commands without submitting before the player sends', () => {
  const submit = vi.fn(async () => true);
  const adventure = { ...view, ruleset: 'pinebrook_d20', suggestions: ['答应护送幼龙', '给幼龙喂食'],
    suggestion_commands: { '答应护送幼龙': { text: '答应护送幼龙', mode: 'in_character' as const, speech_channel: 'local' as const, recipient_ids: [] } } };
  render(<ActionComposer view={adventure} disabled={false} turn={null} pending={null} onSubmit={submit} />);
  fireEvent.change(screen.getByLabelText('选择眼前行动'), { target: { value: '答应护送幼龙' } });
  expect((screen.getByLabelText('你的行动') as HTMLTextAreaElement).value).toBe('答应护送幼龙');
  expect(submit).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: '采取行动' }));
  expect(submit).toHaveBeenCalledWith({ text: '答应护送幼龙', mode: 'in_character', speech_channel: 'local', recipient_ids: [], private_aside: '' });
});
