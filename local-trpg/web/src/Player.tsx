import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import type { CommandFields, PlayerView, PlayerCommand, TurnStatus } from './types';

function speakerName(speaker: string, view: PlayerView): string {
  if (['DM', 'dm', 'narrator', 'Narrator', '主持人'].includes(speaker)) return '主持人';
  if ([view.player.id, view.player.name, 'player', '玩家'].includes(speaker)) return view.player.name;
  return view.npcs.find(npc => npc.id === speaker || npc.name === speaker)?.name ?? speaker;
}

export function Conversation({ view, turn, mode, onModeChange }: { view: PlayerView; turn: TurnStatus | null; mode?: CommandFields['mode']; onModeChange?: (mode: CommandFields['mode']) => void }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);
  const [showLatest, setShowLatest] = useState(false);
  const [localChannel, setLocalChannel] = useState<'story' | 'table'>('story');
  const channel = mode ? mode === 'ooc' ? 'table' : 'story' : localChannel;
  const setChannel = (next: 'story' | 'table') => { setLocalChannel(next); onModeChange?.(next === 'table' ? 'ooc' : 'in_character'); };
  const oocTurn = turn?.mode === 'ooc' || turn?.status === 'completed_ooc';
  const tableTalk = (view.table_talk ?? []).filter(entry => entry.reply_source === 'model');
  const count = view.transcript.length + tableTalk.length;
  const preview = turn?.provisional;
  const replies = preview && turn.status === 'running' && !turn.is_terminal && !turn.cancel_requested &&
    preview.campaign_id === view.campaign_id && preview.turn_id === view.active_turn_id && preview.turn_id === turn.turn_id &&
    preview.epoch === turn.epoch && preview.base_revision === view.revision && preview.base_revision === turn.base_revision
    ? preview.dialogue : [];
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    if (followRef.current) element.scrollTop = element.scrollHeight;
    else setShowLatest(true);
  }, [channel, count, turn?.public_reply, replies.length]);
  useEffect(() => { if (turn) { followRef.current = true; setChannel(oocTurn ? 'table' : 'story'); } }, [turn?.turn_id, oocTurn]);
  const reply = turn?.public_reply && turn.status !== 'committed' && !tableTalk.some(entry => entry.turn_id === turn.turn_id) && !view.transcript.some(entry => entry.turn_id === turn.turn_id && entry.text === turn.public_reply) ? <article className={`turn-reply ${turn.status === 'needs_input' ? 'clarification' : ''}`}>
    <span className="eyebrow">{turn.status === 'completed_ooc' ? turn.reply_source === 'model' ? 'Qwen · 桌外答复' : '旧版桌外答复' : turn.status === 'needs_input' ? '行动需要确认' : '系统消息'}</span><p>{turn.public_reply}</p>
  </article> : null;
  const jumpLatest = () => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
    followRef.current = true; setShowLatest(false);
  };
  return <div className="conversation-wrap">
    <nav className="conversation-channels" aria-label="对话频道"><button type="button" aria-pressed={channel === 'story'} onClick={() => { followRef.current = true; setShowLatest(false); setChannel('story'); }}>冒险记录</button><button type="button" aria-pressed={channel === 'table'} onClick={() => { followRef.current = true; setShowLatest(false); setChannel('table'); }}>桌外对话</button></nav>
    <div className="conversation" ref={scrollRef} role="log" aria-label={channel === 'story' ? '航行记录' : '桌外记录'} aria-live="polite" onScroll={() => {
      const element = scrollRef.current;
      if (element) { followRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 100; if (followRef.current) setShowLatest(false); }
    }}>
      {channel === 'story' && <><div className="chapter-marker"><span />{view.ruleset === 'pinebrook_d20' ? '冒险手记' : '同路号'} · {view.module_title}<span /></div>
      {!count && <p className="quiet empty-transcript">航行记录正在等待你的第一步。</p>}
      {view.transcript.map((entry, index) => {
        const id = entry.speaker_id;
        const name = id === 'dm' ? '主持人' : id === 'player' ? view.player.name :
          view.npcs.find(npc => npc.id === id)?.name ?? speakerName(entry.speaker, view);
        const isDM = id ? id === 'dm' : name === '主持人';
        const isPlayer = id ? id === 'player' : !isDM && name === view.player.name;
        return <article className={`entry ${isDM ? 'dm-entry' : isPlayer ? 'player-entry' : 'npc-entry'}`} key={`${entry.turn_id}-${index}`}>
          <div className="entry-meta"><span className={`speaker-marker ${isDM ? 'dm-marker' : ''}`}>{isDM ? '主' : name.slice(0, 1)}</span><strong>{name}</strong><span className="entry-role">{isDM ? '场景与叙述' : isPlayer ? '你的行动' : view.npcs.find(n => n.name === name)?.role}</span>{view.ruleset !== 'pinebrook_d20' && <span className="entry-tick">第 {entry.tick} 格</span>}</div>
          <div className="entry-text">{entry.text}</div>
        </article>;
      })}
      {replies.length > 0 && <div className="provisional-replies" aria-label="本轮暂存回应">
        <p className="quiet recovery-note">本轮尚未完成，回应暂未写入记录。</p>
        {replies.map(reply => <article className="entry npc-entry provisional-entry" key={`${preview!.epoch}-${reply.event_id}`}>
          <div className="entry-meta"><span className="speaker-marker">{reply.speaker.slice(0, 1)}</span><strong>{reply.speaker}</strong><span className="entry-role">暂存回应</span></div>
          <div className="entry-text">{reply.text}</div>
        </article>)}
      </div>}
      {!oocTurn && reply}
      {!!turn?.result?.checks?.length && <div className="checks" aria-label="本轮检定">{turn.result.checks.map((check, index) => <div className="check-card" key={index}>
        <span className="eyebrow">{check.label || check.skill || '行动检定'}</span>
        <strong>{view.ruleset === 'pinebrook_d20'
          ? `d20${(check.dice?.length ?? 0) > 1 ? ' 优势取高' : ''}：${check.dice?.filter(Number.isFinite).join('、') ?? '骰点'}${typeof check.total === 'number' ? ` → 总计 ${check.total}` : ''}`
          : <>{Array.isArray(check.dice) ? check.dice.filter(Number.isFinite).join(' + ') : '骰点'}{typeof check.total === 'number' ? ` = ${check.total}` : ''}</>}</strong>
        {check.outcome && <span>{check.outcome}</span>}{check.cost && <small>{check.cost}</small>}
      </div>)}</div>}
      {!!turn?.result?.rule_log?.length && <details className="rule-log"><summary>本轮裁定记录</summary>{turn.result.rule_log.map((line, i) => <p key={i}>{line}</p>)}</details>}
      {turn?.result?.degraded && <p className="quiet recovery-note">这一轮使用了简短叙述。已裁定的行动与进度已保留。</p>}
      {view.outcome && <article className="outcome-card">
        <span className="eyebrow">冒险结果</span><h2>{'kind' in view.outcome ? '这一章告一段落' : '航行告一段落'}</h2>
        {!('kind' in view.outcome) && <div className="outcome-grid"><div><strong>{view.outcome.crew_saved}<small> / 6</small></strong><span>船员获救</span></div><div><strong>{view.outcome.cells_saved}<small> / 4</small></strong><span>细胞保全</span></div><div><strong className="outcome-word">{view.outcome.ship_saved ? '保住' : '失去'}</strong><span>同路号</span></div></div>}
        {view.ending_summary?.length ? <div className="ending-explanation">{view.ending_summary.map((text, index) => <p key={index}>{text}</p>)}</div> : <p>本章已结束。你可以向主持人询问已经发生的事情。</p>}
      </article>}
      </>}
      {channel === 'table' && <section className="table-conversation" aria-label="桌外对话">
        <div className="table-heading"><h2>与主持人交谈</h2><span>不进入角色记忆 · 不推进时间</span></div>
        {!tableTalk.length && !reply && <p className="quiet">在下方切换“桌外提问”，输入你自己的问题。可以询问规则、澄清已知情况，或主动请求提示。</p>}
        {tableTalk.map(entry => <div className="table-exchange" key={entry.turn_id}>
          <article className="table-question"><span className="eyebrow">你 · 桌外提问</span><p>{entry.question}</p></article>
          <article className="table-answer"><span className="eyebrow">Qwen · 主持人</span><p>{entry.answer}</p></article>
        </div>)}
        {oocTurn && reply}
      </section>}
      <div className="log-end" aria-hidden="true">· · ·</div>
    </div>
    {showLatest && <button className="latest-button" onClick={jumpLatest}>回到最新记录 ↓</button>}
  </div>;
}

interface ComposerProps {
  mode?: CommandFields['mode']; onModeChange?: (mode: CommandFields['mode']) => void;
  acceptedRequestId?: string | null;
  view: PlayerView; disabled: boolean; turn: TurnStatus | null; pending: { command: PlayerCommand } | null;
  onSubmit: (fields: CommandFields) => Promise<boolean>;
}
export function ActionComposer({ view, disabled, turn, pending, acceptedRequestId, onSubmit, mode: externalMode, onModeChange }: ComposerProps) {
  const [text, setText] = useState(pending?.command.text ?? '');
  const [localMode, setLocalMode] = useState<CommandFields['mode']>(pending?.command.mode ?? 'in_character');
  const mode = externalMode ?? localMode;
  const setMode = (next: CommandFields['mode']) => { setLocalMode(next); onModeChange?.(next); };
  const [channel, setChannel] = useState<CommandFields['speech_channel']>(pending?.command.speech_channel ?? 'local');
  const [recipients, setRecipients] = useState<string[]>(pending?.command.recipient_ids ?? []);
  const [aside, setAside] = useState(pending?.command.private_aside ?? '');
  const [sending, setSending] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const acceptedRef = useRef(acceptedRequestId);
  useEffect(() => {
    if (acceptedRequestId && acceptedRef.current !== acceptedRequestId) { setText(''); setAside(''); }
    acceptedRef.current = acceptedRequestId;
  }, [acceptedRequestId]);
  const endOfStory = Boolean(view.outcome) && mode !== 'ooc';
  const invalidRecipients = mode === 'in_character' && channel === 'private' && recipients.length === 0;
  const send = async (event?: FormEvent) => {
    event?.preventDefault();
    if (disabled || sending || endOfStory || invalidRecipients || !text.trim()) return;
    setSending(true);
    const accepted = await onSubmit({ mode, text, speech_channel: mode === 'ooc' ? 'local' : channel, recipient_ids: mode === 'in_character' && channel === 'private' ? recipients : [], private_aside: mode === 'ooc' ? '' : aside });
    setSending(false);
    if (accepted) { setText(''); setAside(''); }
  };
  useEffect(() => {
    if (pending) { const p = pending.command; setText(p.text); setMode(p.mode); setChannel(p.speech_channel); setRecipients(p.recipient_ids); setAside(p.private_aside); }
  }, [pending?.command.client_request_id]);
  return <form className="composer" onSubmit={event => void send(event)} aria-label="行动输入">
    <div className="composer-heading"><label htmlFor="action-input">{mode === 'ooc' ? '与主持人交谈' : turn?.status === 'needs_input' ? '回应行动确认' : '你的行动'}</label><button type="button" className="guide-question" disabled={disabled || sending} onClick={() => { setMode('ooc'); inputRef.current?.focus(); }}>向主持人提问 ↗</button></div>
    {mode === 'in_character' && !view.outcome && view.ruleset === 'pinebrook_d20' && view.suggestions.length > 0 && <label className="adventure-action-picker"><span>眼前行动</span><select aria-label="选择眼前行动" value="" disabled={disabled || sending} onChange={event => {
      const command = view.suggestion_commands?.[event.target.value];
      setText(command?.text ?? event.target.value);
      if (command) { setChannel(command.speech_channel); setRecipients(command.recipient_ids); }
      inputRef.current?.focus();
    }}><option value="">选择行动后发送，也可直接自由输入</option>{view.suggestions.map(label => <option key={label} value={label}>{label}</option>)}</select></label>}
    {mode === 'in_character' && !view.outcome && view.ruleset !== 'pinebrook_d20' && view.suggestions.length > 0 && <div className="suggestions" aria-label="行动灵感"><span>行动建议</span>{view.suggestions.slice(0, 3).map(suggestion => <button type="button" disabled={disabled || sending} key={suggestion} onClick={() => {
      const command = view.suggestion_commands?.[suggestion];
      setText(command?.text ?? suggestion);
      if (command) { setMode(command.mode); setChannel(command.speech_channel); setRecipients(command.recipient_ids); }
      inputRef.current?.focus();
    }}>{suggestion}<span aria-hidden="true">↗</span></button>)}</div>}
    <div className="input-shell">
      <textarea id="action-input" ref={inputRef} aria-label="你的行动" placeholder={mode === 'ooc' ? '直接输入你的问题，可以连续追问…' : '描述你要做的事，或说出角色的台词…'} rows={2} maxLength={6000} value={text} disabled={disabled || sending || endOfStory} onChange={event => setText(event.target.value)} onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } }} />
      <div className="input-toolbar"><div className="input-controls"><label className="select-field"><span className="sr-only">输入方式</span><select aria-label="输入方式" value={mode} disabled={disabled || sending} onChange={event => setMode(event.target.value as CommandFields['mode'])}><option value="in_character">角色行动</option><option value="ooc">桌外提问</option></select></label>
        {mode === 'in_character' && <label className="select-field"><span className="sr-only">发言范围</span><select aria-label="发言范围" value={channel} disabled={disabled || sending} onChange={event => setChannel(event.target.value as CommandFields['speech_channel'])}><option value="local">现场发言</option><option value="private">{view.ruleset === 'pinebrook_d20' ? '私下交谈' : '私人通话'}</option>{view.ruleset !== 'pinebrook_d20' && <option value="broadcast">工作频道 · 广播</option>}</select></label>}</div>
        <button className="primary send-button" type="submit" disabled={disabled || sending || endOfStory || invalidRecipients || !text.trim()}>{sending ? '正在提交…' : mode === 'ooc' ? '询问主持人' : turn?.status === 'needs_input' ? '回答问题' : '采取行动'}<span aria-hidden="true">↑</span></button>
      </div>
    </div>
    {mode === 'in_character' && channel === 'private' && <fieldset className="recipients" disabled={disabled || sending}><legend>私人通话对象 <span>实际可达范围由当前情境决定</span></legend>{view.npcs.map(npc => <label key={npc.id}><input type="checkbox" value={npc.id} checked={recipients.includes(npc.id)} onChange={event => setRecipients(event.target.checked ? [...recipients, npc.id] : recipients.filter(id => id !== npc.id))} />{npc.name}</label>)}{invalidRecipients && <small>请选择至少一位收件人。</small>}</fieldset>}
    <div className="composer-bottom">{mode === 'in_character' ? <details className="aside-details"><summary>给 DM 的私下说明 <span>可选</span></summary><div><p>只补充你的意图；这段话不会当作角色说出口的话。</p><textarea aria-label="给 DM 的私下说明" placeholder="例如：我想观察他的反应，暂时不透露手里的证据。" rows={2} maxLength={2000} value={aside} disabled={disabled || sending} onChange={event => setAside(event.target.value)} /></div></details> : <span className="quiet">本地 Qwen 回答 · 不推进游戏时间</span>}<span className="shortcut">Ctrl / ⌘ + Enter 发送</span></div>
    {endOfStory && <p className="quiet">这段冒险已经结束。切换“桌外提问”可以继续询问公开信息。</p>}
  </form>;
}

export function PlayerPanel({ view }: { view: PlayerView }) {
  const healthNames: Record<string, string> = { healthy: '状态良好', normal: '状态良好', strained: '轻伤', injured: '受伤', incapacitated: '昏迷' };
  const skillNames: Record<string, string> = view.skill_names ?? { observe: '观察', technical: '技术', social: '交涉', action: '行动' };
  const adventure = view.ruleset === 'pinebrook_d20';
  const ability = view.player.ability;
  return <aside className="player-panel" aria-label="角色与已知信息">
    <div className="mission-title"><span className="eyebrow">当前冒险</span><h1>{view.module_title}</h1><p>{adventure ? '简化 d20 / 角色与旅程' : '同路号 / 船员档案'}</p></div>
    <section className="character-card"><div className="character-avatar" aria-hidden="true">{view.player.name.slice(0, 1)}</div><div><h2>{view.player.name}</h2><p>{view.player.role ?? '随船检修员'}</p></div><span className="health-chip">{healthNames[view.player.health] ?? view.player.health}</span></section>
    {adventure && <section className="character-vitals" aria-label="生命与护甲"><div><span>生命</span><strong>{view.player.hp} / {view.player.max_hp}</strong></div><div><strong>护甲 {view.player.ac}</strong></div></section>}
    {view.player.skills && <details className="panel-details"><summary>角色能力</summary><dl className="skill-list">{Object.entries(view.player.skills).map(([id, bonus]) => <div key={id}><dt>{skillNames[id] ?? id}</dt><dd>{bonus >= 0 ? '+' : ''}{bonus}</dd></div>)}</dl><p className="panel-empty">{adventure ? '技能检定使用d20加对应加值；未列出的技能加值为0。' : '危险且结果不确定时，使用2d6加相关能力裁定。'}</p></details>}
    {adventure && <details className="panel-details" open><summary>攻击与特殊能力</summary><div className="panel-detail-content">{view.player.attacks?.map(attack => <div className="sheet-attack" key={attack.id}><strong>{attack.name}</strong><small>命中 +{attack.bonus} · 伤害 {attack.fixed_damage ?? `${attack.dice}d6 + ${attack.damage_bonus ?? 0}`}</small></div>)}{ability && <div className="sheet-ability"><strong>{ability.name}</strong><p>{ability.description ?? `恢复${ability.heal}点生命，不超过生命上限。`}</p><small>{ability.uses === null ? '依照触发条件自动生效' : `剩余 ${ability.remaining ?? ability.uses} / ${ability.uses} 次`}</small></div>}{typeof view.player.ammo === 'number' && <p className="quiet">箭矢：{view.player.ammo}</p>}</div></details>}
    <section className="location-section"><div className="panel-heading"><h2>{view.outcome ? '最后行动地点' : '所在地点'}</h2><span aria-hidden="true">⌖</span></div><p className="current-location">{view.player.location}</p>{adventure && <p className="quiet">眼前相邻去向；能否通过取决于现场情况。</p>}<div className="known-locations">{view.locations.map(location => <span key={location.id} className={location.id === view.player.location_id ? 'here' : ''}>{location.name}</span>)}</div></section>
    {adventure && view.combat && <section className="encounter-panel" aria-label="当前战斗"><div className="panel-heading"><h2>当前战斗</h2><span>第 {view.combat.round} 轮</span></div>{view.combat.enemies.map(enemy => <div className="enemy-row" key={enemy.id}><strong>{enemy.name}</strong><span>{enemy.condition}</span></div>)}<p className="quiet">写明攻击目标、治疗或撤退。普通提问不会触发敌人行动。</p></section>}
    <section className="clock-section" aria-label={adventure ? '行动规则' : '危机时钟'}>{view.clocks.map((clock, index) => <div className="crisis-clock" key={`${clock.label}-${index}`}><div className="clock-label"><span>{clock.label}</span><strong>{clock.value}<small> / {clock.limit}</small></strong></div><div className={`clock-track ${clock.value / clock.limit >= .75 ? 'urgent' : ''}`} role="progressbar" aria-label={clock.label} aria-valuenow={clock.value} aria-valuemin={0} aria-valuemax={Math.max(clock.limit, clock.value)}><span style={{ width: `${Math.max(0, Math.min(100, clock.value / Math.max(clock.limit, 1) * 100))}%` }} /></div></div>)}<p className="clock-note">{adventure ? '实质行动按现场规则结算，普通问答不推进战斗轮。' : '调查、交涉和实际工作推进时间，普通问答不耗时。'}</p></section>
    <details className="panel-details" open><summary><span>随身物品</span><span className="count">{view.player.inventory.length}</span></summary><div className="panel-detail-content">{view.player.inventory.length ? <ul className="inventory-list">{view.player.inventory.map(item => <li key={item.id}><span aria-hidden="true">◇</span>{item.name}</li>)}</ul> : <p className="panel-empty">暂时没有随身物品。</p>}</div></details>
    <details className="panel-details" open><summary><span>已获证据</span><span className="count">{view.evidence.length}</span></summary><div className="panel-detail-content">{view.evidence.length ? view.evidence.map(evidence => <details className="evidence-card" key={evidence.id}><summary>{evidence.title}</summary><p>{evidence.text}</p>{evidence.source && <small>来源：{view.npcs.find(npc => npc.id === evidence.source)?.name ?? evidence.source}</small>}</details>) : <p className="panel-empty">调查现场，收集你的第一条线索。</p>}</div></details>
    {view.party && <details className="panel-details" open><summary>同行队伍</summary><div className="panel-detail-content">{view.party.filter(member => member.id !== 'player').map(member => <div className="party-row" key={member.id}><strong>{member.name}</strong><span>{member.role} · 生命 {member.hp} / {member.max_hp}</span><small>{member.ability.name}{member.ability.uses !== null ? ` · 剩余${member.ability.remaining}次` : ''}</small></div>)}</div></details>}
    <details className="panel-details crew-details" open><summary><span>{adventure ? '眼前人物' : '船员名册'}</span><span className="count">{view.npcs.length}</span></summary><div className="panel-detail-content">{view.npcs.map(npc => <div className="crew-row" key={npc.id}><span className="crew-initial" aria-hidden="true">{npc.name.slice(0, 1)}</span><div><strong>{npc.name}</strong><small>{npc.role}</small></div><span className={`presence ${npc.present ? 'present' : ''}`}>{npc.present ? '在场' : '不在场'}</span></div>)}</div></details>
  </aside>;
}
