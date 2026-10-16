import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { downloadFile, errorMessage } from './api';
import type { ClientState } from './client';
import { GameClient } from './client';
import type { SaveInfo } from './types';

export function Dialog({ title, children, onClose, wide = false }: { title: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current; dialog?.showModal(); return () => dialog?.close(); }, []);
  return <dialog className={`dialog ${wide ? 'dialog-wide' : ''}`} ref={ref} aria-label={title} onCancel={event => { event.preventDefault(); onClose(); }} onClick={event => { if (event.target === event.currentTarget) { const rect = event.currentTarget.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose(); } }}>
    <div className="dialog-heading"><h2>{title}</h2><button className="icon-button" type="button" aria-label="关闭" onClick={onClose}>×</button></div>{children}
  </dialog>;
}

export function NewCampaignDialog({ client, state, onClose, initialModuleId }: { client: GameClient; state: ClientState; onClose: () => void; initialModuleId?: string }) {
  const [name, setName] = useState('');
  const [moduleId, setModuleId] = useState(initialModuleId || state.modules[0]?.module_id || '');
  const [characterId, setCharacterId] = useState('noorah');
  const selected = state.modules.find(module => module.module_id === moduleId);
  const character = selected?.characters?.find(card => card.id === characterId) ?? selected?.characters?.[0];
  return <Dialog title="开始一段新冒险" onClose={onClose}><form onSubmit={async event => { event.preventDefault(); if (await client.createCampaign(moduleId, name, character?.id)) onClose(); }} className="dialog-content">
    <p className="quiet">{selected?.description}</p>
    <label className="form-field">角色名字<input autoFocus required maxLength={30} value={name} placeholder="你希望大家怎样称呼你？" onChange={event => setName(event.target.value)} disabled={state.loading} /></label>
    <fieldset className="module-options"><legend>选择冒险</legend>{state.modules.map(module => <label className={`module-option ${moduleId === module.module_id ? 'selected' : ''}`} key={module.module_id}><input type="radio" name="module" value={module.module_id} checked={moduleId === module.module_id} onChange={() => setModuleId(module.module_id)} /><span><strong>{module.title}</strong><small>{module.description}</small></span></label>)}</fieldset>
    {!!selected?.characters?.length && <fieldset className="module-options"><legend>选择角色卡</legend>{selected.characters.map(card => <label className={`module-option ${character?.id === card.id ? 'selected' : ''}`} key={card.id}><input type="radio" name="character" value={card.id} checked={character?.id === card.id} onChange={() => setCharacterId(card.id)} /><span><strong>{card.role}</strong><small>{card.name} · 生命 {card.hp} · 护甲 {card.ac}</small><small>{card.attacks.map(attack => attack.name).join(' / ')} · {card.ability.name}</small></span></label>)}</fieldset>}
    {selected?.adaptations && <details className="adventure-notes"><summary>单人版规则与原模组差异</summary><ul>{selected.adaptations.map(note => <li key={note}>{note}</li>)}</ul>{selected.source_url && <a href={selected.source_url} target="_blank" rel="noreferrer">官方资料入口</a>}</details>}
    {!state.modules.length && <p role="alert">冒险资料尚未载入。请关闭窗口并重试连接。</p>}
    {state.error && <p className="inline-error" role="alert">{state.error}</p>}
    <div className="dialog-actions"><button className="secondary" type="button" onClick={onClose}>取消</button><button className="primary" disabled={state.loading || !name.trim() || !moduleId}>{state.loading ? '准备冒险…' : '进入冒险'}<span aria-hidden="true">→</span></button></div>
  </form></Dialog>;
}

function dateLabel(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '最近的航程' : date.toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function LibraryDialog({ client, state, onClose, onNew }: { client: GameClient; state: ClientState; onClose: () => void; onNew: () => void }) {
  const [tab, setTab] = useState<'campaigns' | 'saves' | 'files'>(state.view ? 'saves' : 'campaigns');
  const [saves, setSaves] = useState<SaveInfo[]>([]);
  const [name, setName] = useState(state.view ? `第 ${state.view.tick} 格 · ${state.view.player.location}` : '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [spoilerConfirmed, setSpoilerConfirmed] = useState(false);
  const [savesLoading, setSavesLoading] = useState(Boolean(state.view));
  const mutationBlocked = state.loading || state.submitting || Boolean(state.pending) || busy;
  useEffect(() => {
    let current = true;
    void client.refreshLibrary();
    if (state.view) void client.loadSaves().then(data => { if (current) setSaves(data); }).catch(err => { if (current) setError(errorMessage(err)); }).finally(() => { if (current) setSavesLoading(false); });
    return () => { current = false; };
  }, [client, state.view?.campaign_id]);
  const task = async (work: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await work(); } catch (err) { setError(errorMessage(err)); } finally { setBusy(false); }
  };
  const importFile = async () => {
    if (!file) return;
    if (file.size > 50 * 1024 * 1024) { setError('备份超过 50 MB，请选择本应用导出的 JSON 备份。'); return; }
    setBusy(true); setError('');
    try {
      let data: unknown;
      try { data = JSON.parse(await file.text()); } catch { setError('无法读取这个 JSON 文件，请选择完整的战役备份。'); return; }
      if (await client.importBackup(data)) onClose();
    } finally { setBusy(false); }
  };
  return <Dialog title="战役与存档" onClose={onClose} wide><div className="dialog-tabs" role="tablist" aria-label="存档管理"><button role="tab" aria-selected={tab === 'saves'} onClick={() => setTab('saves')}>这一程的存档</button><button role="tab" aria-selected={tab === 'campaigns'} onClick={() => setTab('campaigns')}>所有战役</button><button role="tab" aria-selected={tab === 'files'} onClick={() => setTab('files')}>导入与导出</button></div>
    <div className="dialog-content library-content" role="tabpanel">
      {tab === 'saves' && (state.view ? <><form className="save-form" onSubmit={event => { event.preventDefault(); void task(async () => { if (await client.save(name)) { setSaves(await client.loadSaves()); } }); }}><label className="form-field">存档名称<input value={name} maxLength={80} onChange={event => setName(event.target.value)} placeholder="例：进入动力舱前" disabled={busy} /></label><button className="primary" disabled={busy || !name.trim()}>保存进度</button></form><p className="quiet">行动进行中也可以存档，保存的是上一已完成回合。每次载入都会开启一个新分支。</p>
        {state.notice && <p className="inline-success" role="status">{state.notice}</p>}
        <div className="saved-list">{savesLoading ? <p className="empty-list">正在查找存档…</p> : !saves.length ? <p className="empty-list">还没有手动存档。输入名称即可保存当前进度。</p> : saves.map(save => <div className="saved-row" key={save.save_id}><span className="save-symbol" aria-hidden="true">◇</span><div><strong>{save.name}</strong><small>第 {save.tick} 格</small></div><button className="secondary" disabled={mutationBlocked} onClick={() => void task(async () => { if (await client.fork(save.save_id)) onClose(); })}>从此处继续</button></div>)}</div>
      </> : <p className="empty-list">先开始或载入一段冒险，再为它创建存档。</p>)}
      {tab === 'campaigns' && <><div className="library-intro"><p className="quiet">每段冒险都保留各自的进度。</p><button className="secondary" disabled={mutationBlocked} onClick={onNew}>＋ 新建战役</button></div><div className="saved-list">{!state.campaigns.length ? <p className="empty-list">你还没有开始冒险。</p> : state.campaigns.map(campaign => <div className="saved-row" key={campaign.campaign_id}><span className="save-symbol" aria-hidden="true">↳</span><div><strong>{campaign.player_name}{campaign.campaign_id === state.view?.campaign_id && <span className="current-badge">当前</span>}</strong><small>{state.modules.find(m => m.module_id === campaign.module_id)?.title ?? '备用冷却'} · 第 {campaign.tick} 格 · {dateLabel(campaign.updated_at)}</small></div><button className="secondary" disabled={mutationBlocked || campaign.campaign_id === state.view?.campaign_id} onClick={() => void task(async () => { if (await client.openCampaign(campaign.campaign_id)) onClose(); })}>继续</button></div>)}</div></>}
      {tab === 'files' && <div className="file-sections"><section><h3>分享航行记录</h3><p>导出玩家已见的文字记录，便于重读或分享。不会包含给 DM 的私下说明。</p><button className="secondary" disabled={!state.view || busy} onClick={() => void task(async () => { await downloadFile(`/api/campaigns/${encodeURIComponent(state.view!.campaign_id)}/transcript`, `${state.view!.module_title}-航行记录.md`); })}>下载文字记录</button></section>
        <section className="backup-section"><h3>完整战役备份 <span>含剧透</span></h3><p>包含主持人资料、角色私有状态和隐藏真相。用于本人备份，请勿作为无剧透记录分享。</p><label className="checkbox-label"><input type="checkbox" checked={spoilerConfirmed} onChange={event => setSpoilerConfirmed(event.target.checked)} />我了解完整备份含剧透</label><button className="secondary" disabled={!state.view || busy || !spoilerConfirmed} onClick={() => void task(async () => { await downloadFile(`/api/campaigns/${encodeURIComponent(state.view!.campaign_id)}/export`, `${state.view!.module_title}-完整备份-含剧透.json`); })}>下载完整备份</button></section>
        <section><h3>从备份恢复</h3><p>选择此前导出的 JSON 备份，将建立一段新战役，不会覆盖已有进度。</p><label className="file-input-label"><span>选择 JSON 备份</span><input type="file" accept=".json,application/json" aria-label="选择 JSON 备份" disabled={mutationBlocked} onChange={event => setFile(event.target.files?.[0] ?? null)} /></label><button className="primary" disabled={!file || mutationBlocked} onClick={() => void importFile()}>{busy ? '正在恢复…' : '导入为新战役'}</button></section></div>}
      {(error || state.error) && <p className="inline-error" role="alert">{error || state.error}</p>}
      {state.pending && <p className="quiet">有一条行动尚待确认，请先返回游戏检查它的状态，再切换战役。</p>}
    </div>
  </Dialog>;
}

export function HelpDialog({ onClose, adventure = false }: { onClose: () => void; adventure?: boolean }) {
  return <Dialog title="操作说明" onClose={onClose}><div className="dialog-content help-content"><p className="help-lead">阅读冒险记录，在行动框中描述你要做的事。主持人将裁定结果。</p><ol><li><strong>从一个具体行动开始</strong><p>例如“查看眼前物件上的痕迹”，或“我问队长，刚才发生了什么”。你也可以提出建议之外的办法。</p></li><li><strong>决定谁能听见</strong><p>现场发言面向能听见你的人；私人通话选择对象；工作频道广播给船员。实际传达仍取决于位置和通讯条件。</p></li><li><strong>把心里话放在合适的位置</strong><p>展开“给 DM 的私下说明”补充试探、隐瞒或观察的意图。把实际说出口的话写在主输入框里。</p></li><li><strong>行动会留下后果</strong><p>时间、物品、关系和线索随裁定变化。人物可能误解或说谎，听到的说法不等于已核实的事实。</p></li></ol><div className="help-note"><strong>不确定时，随时问。</strong><p>切换“桌外提问”询问公开规则，不推进游戏时间。主持人要求澄清时，直接回答问题即可。</p></div><p className="quiet">本地主持人可能需要一些时间。生成过程中可以继续阅读、保存上一回合，或请求取消；断线后刷新即可查询已有进度。不要在等待时重复发送同一个行动。</p>{adventure && <p>技能与攻击使用d20。你控制一名角色，三位同伴会在战斗轮中以普通攻击协助；治疗和有限能力需要明确提出。</p>}<button className="primary full-width" onClick={onClose}>明白了，回到冒险 →</button></div></Dialog>;
}
