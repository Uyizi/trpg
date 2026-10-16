import { useEffect, useState, useSyncExternalStore } from 'react';
import { actionBlocked, GameClient } from './client';
import { HelpDialog, LibraryDialog, NewCampaignDialog } from './Dialogs';
import { ActionComposer, Conversation, PlayerPanel } from './Player';
import type { TurnStatus } from './types';
import { ModelAvailability } from './ModelAvailability';

const client = new GameClient();
const stageLabels: Record<string, string> = {
  '理解冒险行动': '理解你的行动', 'Qwen扮演现场人物': '眼前的人物正在回应', '裁定冒险行动': '裁定行动与后果', 'Qwen叙述实际结果': '描述眼前发生的事', '提交冒险进度': '记录这一轮进展', 'Qwen回答桌外问题': '主持人正在回答',
  queued: '这一轮正在排队', warmup: '主持人准备中，首次启动可能需要几分钟', loading_model: '主持人准备中，首次启动可能需要几分钟',
  interpreting: '理解你的行动', interpretation: '理解你的行动', input: '理解你的行动', planning: '梳理当前局势',
  resolving: '裁定行动与后果', rules: '裁定行动与后果', adjudicating: '裁定行动与后果',
  npcs: '船员正在回应', npc: '船员正在回应', npc_response: '船员正在回应',
  narrating: '写下眼前发生的事', narration: '写下眼前发生的事', committing: '记录这一轮进展',
};
function turnLabel(turn: TurnStatus | null) {
  if (!turn) return '正在查询这一轮的进度';
  if (turn.cancel_requested && !turn.is_terminal) return '正在取消，等待这一轮安全停止';
  return stageLabels[turn.stage] ?? (turn.status === 'queued' ? '这一轮正在排队' : '主持人正在处理这一轮');
}

export default function App() {
  const state = useSyncExternalStore(client.subscribe, client.getSnapshot);
  const [dialog, setDialog] = useState<'new' | 'library' | 'help' | null>(null);
  const [newModuleId, setNewModuleId] = useState('');
  const [showPanel, setShowPanel] = useState(false);
  const [inputMode, setInputMode] = useState<'in_character' | 'ooc'>('in_character');
  useEffect(() => { void client.start(); return () => client.dispose(); }, []);
  const { view, turn } = state;
  const adventure = view?.ruleset === 'pinebrook_d20';
  useEffect(() => { if (view) setInputMode('in_character'); }, [view?.campaign_id]);
  const running = Boolean(state.turnId && (!turn || !turn.is_terminal));
  const recoverable = turn?.status === 'recoverable';
  const switchBlocked = state.submitting || Boolean(state.pending);
  return <div className={`app ${view ? 'has-campaign' : 'welcome-app'}`}>
    <a className="skip-link" href="#main-content">跳到冒险记录</a>
    <header className="topbar"><a className="brand" href="#main-content" aria-label="冒险手记"><span className="brand-name">{view && !adventure ? '同路号' : '冒险手记'}</span><span className="brand-description">本地文字跑团</span></a><div className="topbar-right"><ModelAvailability health={state.health} warming={state.warming} onWarmup={() => void client.warmupModel()} /><nav aria-label="主导航"><button className="topbar-button" onClick={() => setDialog('library')}>战役与存档</button><button className="topbar-button help-button" onClick={() => setDialog('help')}>操作说明</button></nav></div></header>
    {(state.health?.mock || view?.mock) && <div className="mock-banner" role="status">测试模式 · 当前使用固定测试响应，不代表真实模型的游戏体验。</div>}
    {!state.storageAvailable && <div className="message-banner warning" role="status">浏览器无法保存重连信息。进度仍保存在本地服务中，重新打开页面后请从“战役与存档”继续。</div>}
    {state.error && <div className="message-banner error" role="alert"><span>{state.error}</span><button onClick={() => state.view ? void client.refreshCampaign() : void client.start()}>重新连接</button><button aria-label="关闭提示" onClick={client.clearMessage}>×</button></div>}
    {state.notice && !dialog && <div className="message-banner success" role="status"><span>{state.notice}</span><button aria-label="关闭提示" onClick={client.clearMessage}>×</button></div>}
    {view ? <div className={`game-layout ${showPanel ? 'show-panel' : ''}`}>
      <div className="mobile-panel-toggle"><button aria-expanded={showPanel} onClick={() => setShowPanel(!showPanel)}>{showPanel ? '收起角色与线索 ↑' : `${view.player.location} · 角色与线索 ↓`}</button><span>{adventure ? '简化 d20' : `第 ${view.tick} 格`}</span></div>
      <PlayerPanel view={view} />
      <main className="game-main" id="main-content"><div className="log-header"><div><span className="log-kicker">{view.module_title}</span><h2>{adventure ? '冒险记录' : '航行记录'}</h2></div><div className="log-header-meta"><span className="location-label">{view.outcome ? '本章已结束' : view.player.location}</span>{!adventure && <span className="turn-count">第 <strong>{view.tick}</strong> 格</span>}<button className="text-button new-campaign-button" disabled={switchBlocked} onClick={() => setDialog('new')}>新冒险</button></div></div>
        <Conversation key={view.campaign_id} view={view} turn={turn} mode={inputMode} onModeChange={setInputMode} />
        {(running || recoverable || state.pending || state.pollError) && <div className={`turn-status ${recoverable || state.pending ? 'needs-attention' : ''}`} role="status" aria-live="polite">
          {state.pending ? <><div><strong>{state.submitting ? '正在提交行动…' : '这条行动的接收状态尚未确认'}</strong><p>{state.submitting ? '请稍候，原有记录仍可继续阅读。' : '原文已保留。重试会确认同一条行动，不会另发一次。'}</p></div><button className="secondary" disabled={state.submitting} onClick={() => void client.retrySubmission()}>重试这条行动</button></> : recoverable ? <><div><strong>这一轮可以恢复</strong><p>继续原来的行动，或取消后重新考虑。</p></div><div className="status-actions"><button className="secondary" disabled={state.submitting} onClick={() => void client.cancelTurn()}>取消这一轮</button><button className="primary" disabled={state.submitting} onClick={() => void client.resumeTurn()}>恢复这一轮</button></div></> : <><span className="activity-dot" aria-hidden="true" /><div><strong>{state.pollError ? '连接中断，正在查询原来的行动' : turnLabel(turn)}</strong><p>{state.pollError ?? '可以继续阅读记录。等待本身不会推进游戏时间。'}</p></div><button className="text-button" disabled={state.submitting || Boolean(turn?.cancel_requested)} onClick={() => void client.cancelTurn()}>{turn?.cancel_requested ? '取消中…' : '取消生成'}</button></>}
        </div>}
        <ActionComposer key={`composer-${view.campaign_id}`} view={view} mode={inputMode} onModeChange={setInputMode} disabled={actionBlocked(state)} turn={turn} pending={state.pending} acceptedRequestId={state.acceptedRequestId} onSubmit={fields => client.submit(fields)} />
      </main>
    </div> : <main className="welcome" id="main-content">
      <div className="welcome-heading"><span className="eyebrow">单人文字冒险 / 进度保存在本地</span><h1>开始一段冒险</h1><p>选择故事与角色。观察现场，与人物交谈，用文字决定你的行动。</p></div>
      {state.modules.map(module => <section className="adventure-card" aria-label={module.title} key={module.module_id}>
        <div className="adventure-card-heading"><span className="adventure-type">{module.ruleset === 'pinebrook_d20' ? '奇幻 / 官方短模组的单人适配' : '科幻 / 船舱救援'}</span><span className="adventure-format">{module.ruleset === 'pinebrook_d20' ? '1 位玩家 · 3 位同伴' : '1 位玩家 · 5 位船员'}</span></div>
        <h2>{module.title}</h2><p className="adventure-description">{module.description}</p>
        <div className="adventure-setup"><span>你的角色</span><p>{module.characters ? '战士、游荡者、法师、牧师，四张预设卡任选一张' : '临时随船检修员'}</p><span>行动方式</span><p>自由文字输入，由本地主持人和本场规则结算</p></div>
        <div className="adventure-card-footer"><span>创建角色后开始，进度保存在本地。</span><button className="primary" disabled={state.loading} onClick={() => { setNewModuleId(module.module_id); setDialog('new'); }}>开始冒险<span aria-hidden="true">→</span></button></div>
      </section>)}
      {state.loading && <p className="quiet">正在读取本地冒险…</p>}
      <section className="continue-adventure"><div><h2>继续上次的冒险</h2><p>打开已有战役，或从手动存档创建分支。</p></div><button className="secondary" onClick={() => setDialog('library')}>继续已有航程</button></section>
      <footer className="welcome-footer"><span>首次游玩可以先了解行动、通话与存档。</span><button className="text-button" onClick={() => setDialog('help')}>查看操作说明</button></footer>
    </main>}

    {dialog === 'new' && <NewCampaignDialog initialModuleId={newModuleId} client={client} state={state} onClose={() => setDialog(null)} />}
    {dialog === 'library' && <LibraryDialog client={client} state={state} onClose={() => setDialog(null)} onNew={() => setDialog('new')} />}
    {dialog === 'help' && <HelpDialog adventure={adventure} onClose={() => setDialog(null)} />}
  </div>;
}
