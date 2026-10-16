/* Run with Windows Node: node.exe <Windows path to this file>.
 * Uses installed Windows Chrome and a throwaway profile. Default: deterministic OOC/MOVE.
 * Explicit --real: one real private doctor conversation, evidence, save, and fork.
 * Explicit --opening: click the actual look-around/broadcast suggestions and require real NPC replies.
 * --describe prints the selected plan without launching Chrome or sending requests.
 * No npm/browser downloads are needed. Schedule --real exclusively with the GPU owner.
 */
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const assert = require('node:assert/strict');
const flags = new Set(process.argv.slice(2));
for (const flag of flags) if (!['--real', '--opening', '--describe'].includes(flag)) throw new Error(`Unknown argument: ${flag}`);
const openingMode = flags.has('--opening');
if (openingMode && flags.has('--real')) throw new Error('Choose --opening or --real');
const realMode = flags.has('--real') || openingMode;
const runId = new Date().toISOString().replace(/[-:.]/g, '');
const artifactPrefix = openingMode ? `browser-opening-${runId}` : realMode ? `browser-real-${runId}` : 'browser-smoke';
const realCommand = '方榆，细胞如何保冷与转存？请把货物说明告诉我。';
const privateCanary = 'PRIVATE_UI_CANARY_20261009';
if (flags.has('--describe')) {
  if (openingMode) {
    console.log(JSON.stringify({ mode: 'real-opening', executes: false, steps: ['Create isolated campaign through browser UI', 'Click look-around suggestion and submit; require visible room and zero time cost', 'Click work-channel suggestion; require broadcast selection and actual POST', 'Require original local model to return non-degraded NPC dialogue; retain raw result and screenshot'], request_limit: 2 }, null, 2));
    process.exit(0);
  }
  console.log(JSON.stringify({ mode: realMode ? 'real-private-evidence' : 'deterministic', executes: false, steps: realMode ? ['Require ready non-mock idle runner', 'Create a new dedicated campaign through UI', 'Choose private channel and doctor checkbox', `Fill actual DM aside with ${privateCanary}`, `Submit exactly once: ${realCommand}`, 'Audit actual POST.private_aside; reject canary in any public DTO, share transcript, player log or NPC content', 'Wait up to 180 seconds for the real terminal status; preserve failures', 'Require committed non-degraded result and actual E05 in both API view and visible UI', 'Save and fork through UI, verify E05 persists, screenshot', 'Reload without another action; record console and request audit'] : ['Create campaign', 'OOC', 'Zero-cost MOVE', 'Save and fork', 'Reload', 'Mobile screenshots'], request_limit: realMode ? 1 : 2 }, null, 2));
  process.exit(0);
}
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'artifacts');
const base = 'http://127.0.0.1:8768';
const debugPort = realMode ? 19369 : 19368;
const chrome = process.env.TRPG_CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const profile = fs.mkdtempSync(path.join(os.tmpdir(), realMode ? 'local-trpg-ui-real-' : 'local-trpg-ui-smoke-'));
const report = { run_id: runId, artifact_prefix: artifactPrefix, mode: openingMode ? 'real-opening' : realMode ? 'real-private-evidence' : 'deterministic', started_at: new Date().toISOString(), url: base, browser: '', node: process.version, profile_isolated: true, proxy_disabled: true, steps: [], errors: [], warnings: [], failed_requests: [], posted_turns: [], campaigns_created: [], screenshots: [] };
fs.mkdirSync(output, { recursive: true });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const api = async route => {
  const r = await fetch(base + route);
  if (!r.ok) throw new Error(`GET ${route}: ${r.status}`);
  const value = await r.json();
  if (realMode) {
    const leaked = JSON.stringify(value).includes(privateCanary);
    (report.public_dto_canary_checks ||= []).push({ route, contains_canary: leaked });
    if (leaked) { report.private_aside_leak = { route, actual_public_body: value }; throw new Error(`Private aside canary leaked into public DTO: ${route}`); }
  }
  return value;
};
let chromeProcess, socket, commandId = 0;
const pending = new Map();
const step = (name, details = {}) => { report.steps.push({ name, ...details }); console.log(JSON.stringify({ step: name, ...details })); };
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++commandId;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timed out: ${method}`)); }, 15000);
    pending.set(id, { resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); } });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
async function waitFor(expression, label, timeout = 20000) {
  const until = Date.now() + timeout;
  let last;
  while (Date.now() < until) {
    try { const result = await evaluate(`(() => { const value = (${expression}); return value instanceof Element ? true : value; })()`); if (result) return result; } catch (error) { last = error; }
    await delay(180);
  }
  throw new Error(`Waiting for ${label} failed${last ? ': ' + last.message : ''}`);
}
async function clickExpression(expression, label) {
  const point = await evaluate(`(() => { const el = ${expression}; if (!el) throw new Error(${JSON.stringify('Missing element: ' + label)}); if (el.disabled) throw new Error(${JSON.stringify('Disabled: ' + label)}); el.scrollIntoView({block:'center',inline:'center'}); const r=el.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2}; })()`);
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', button: 'left', clickCount: 1, ...point });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', button: 'left', clickCount: 1, ...point });
}
async function clickButton(label) {
  const text = JSON.stringify(label);
  await clickExpression(`Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim().replace(/[→↑↗＋]/g,'').trim() === ${text})`, label);
}
async function fill(selector, value) {
  await evaluate(`(() => { const el=document.querySelector(${JSON.stringify(selector)}); if(!el)throw new Error('Missing input'); el.focus(); el.select(); })()`);
  await send('Input.insertText', { text: value });
}
async function session() { return evaluate(`JSON.parse(localStorage.getItem('local-trpg.session.v1') || 'null')`); }
async function screenshot(name, full = false) {
  const options = { format: 'png', captureBeyondViewport: full, fromSurface: true };
  if (full) {
    const metrics = await send('Page.getLayoutMetrics');
    const size = metrics.cssContentSize;
    options.clip = { x: 0, y: 0, width: size.width, height: Math.min(size.height, 7000), scale: 1 };
  }
  const image = await send('Page.captureScreenshot', options);
  const filename = `${artifactPrefix}-${name}.png`;
  fs.writeFileSync(path.join(output, filename), Buffer.from(image.data, 'base64'));
  report.screenshots.push(filename);
}
async function waitTurn(campaignId, previousTurnId, timeoutMs = 25000) {
  const saved = await waitFor(`(() => {const s=JSON.parse(localStorage.getItem('local-trpg.session.v1')||'null');return s && s.turnId && s.turnId!==${JSON.stringify(previousTurnId)} && !s.pending ? s : false;})()`, 'new turn receipt');
  const started = Date.now();
  const until = started + timeoutMs;
  let previousStage = '';
  let lastProgress = started;
  while (Date.now() < until) {
    const result = await api(`/api/campaigns/${campaignId}/turns/${saved.turnId}`);
    const stage = `${result.status}:${result.stage}`;
    report.last_turn = { campaign_id: campaignId, turn_id: saved.turnId, status: result.status, stage: result.stage, elapsed_ms: Date.now() - started };
    if (realMode && result.provisional?.dialogue?.length && !report.provisional_api) {
      report.provisional_api = { elapsed_ms: Date.now() - started, status: result.status,
        revision: result.revision, has_final_result: !!result.result,
        dialogue: result.provisional.dialogue };
      console.log(JSON.stringify({ provisional_api: report.provisional_api }));
    }
    if (realMode && !report.provisional_dom) {
      const visible = await evaluate(`Array.from(document.querySelectorAll('.provisional-entry .entry-text')).map(el=>el.innerText)`);
      if (visible.length) {
        report.provisional_dom = { elapsed_ms: Date.now() - started, dialogue: visible };
        await screenshot('provisional-reply');
      }
    }
    if (stage !== previousStage || Date.now() - lastProgress >= 10000) {
      console.log(JSON.stringify({ turn_progress: report.last_turn }));
      previousStage = stage; lastProgress = Date.now();
    }
    if (result.is_terminal) {
      if (realMode) {
        report.real_terminal = { turn_id: result.turn_id, status: result.status, stage: result.stage, is_terminal: result.is_terminal, revision: result.revision, public_reply: result.public_reply, degraded: result.result?.degraded ?? null };
        fs.writeFileSync(path.join(output, `${artifactPrefix}-turn.json`), JSON.stringify(result, null, 2) + '\n');
      }
      await delay(1300); return result;
    }
    await delay(realMode ? 1000 : 300);
  }
  throw new Error(`Turn did not reach terminal state within ${timeoutMs} ms; it was not cancelled or resubmitted`);
}
async function realOpening(first) {
  await clickButton('查看这里有什么可以调查的');
  await clickButton('采取行动');
  const looked = await waitTurn(first.campaignId, null);
  assert.equal(looked.status, 'committed');
  assert.equal(looked.result.view.tick, 0);
  assert.ok(looked.public_reply.includes('这里可见'));
  report.look_around = looked;
  await clickButton('使用工作频道询问当前情况');
  assert.equal(await evaluate(`document.querySelector('select[aria-label="发言范围"]').value`), 'broadcast');
  const health = await api('/api/health');
  assert.equal(health.model_available, true);
  assert.equal(health.model_busy, false);
  assert.equal(health.mock, false);
  await clickButton('采取行动');
  const turn = await waitTurn(first.campaignId, looked.turn_id, 180000);
  assert.equal(turn.status, 'committed');
  assert.equal(turn.result.degraded, false);
  assert.ok(turn.result.dialogue.length > 0, 'A real NPC must answer the broadcast');
  assert.equal(turn.result.view.tick, 1);
  assert.equal(report.posted_turns.length, 2);
  assert.equal(report.posted_turns[1].text, '使用工作频道询问当前情况');
  assert.equal(report.posted_turns[1].speech_channel, 'broadcast');
  assert.deepEqual(report.posted_turns[1].recipient_ids, []);
  await waitFor(`document.querySelector('.npc-entry .entry-text')`, 'real NPC reply rendered');
  report.rendered_dialogue = await evaluate(`Array.from(document.querySelectorAll('.npc-entry .entry-text')).map(el=>el.innerText)`);
  assert.ok(report.rendered_dialogue.includes(turn.result.dialogue[0].text));
  assert.equal(await evaluate(`document.querySelectorAll('.player-entry').length`), 2, 'Both submitted actions must remain in the visible log');
  for (const line of turn.result.dialogue) assert.ok(!/一切正常|没有发现异常|未发现异常/.test(line.text), 'Status may not infer room safety from character health');
  const world = JSON.parse(fs.readFileSync(path.join(root, 'modules/reserve_cooling/world.json'), 'utf8'));
  for (const line of turn.result.dialogue) {
    const npc = world.actors[line.speaker_id];
    assert.ok(npc, 'Reply must identify the actual NPC');
    // This isolated opening has no movement/work authorizations or clock triggers.
    const location = world.locations[npc.location_id].name;
    assert.ok(line.text.includes(location), `${line.speaker} must report their own location: ${location}`);
  }
  await screenshot('reply', true);
  step('actual opening buttons reached original model and rendered NPC replies', {campaign_id: first.campaignId, dialogue: turn.result.dialogue});
  // Optional read-only reproduction of restoring a historical user turn. This
  // isolated browser changes only its own local session and submits no action.
  const historicalCampaign = process.env.TRPG_RECOVERY_CAMPAIGN;
  const historicalTurn = process.env.TRPG_RECOVERY_TURN;
  if (historicalCampaign && historicalTurn) {
    const before = await api(`/api/campaigns/${historicalCampaign}`);
    const posts = report.posted_turns.length;
    await evaluate(`localStorage.setItem('local-trpg.session.v1', ${JSON.stringify(JSON.stringify({campaignId: historicalCampaign, turnId: historicalTurn, pending: null}))})`);
    await send('Page.reload', { ignoreCache: true });
    await waitFor(`document.querySelector('textarea[aria-label="你的行动"]') && !document.querySelector('textarea[aria-label="你的行动"]').disabled`, 'historical campaign after status restoration');
    await clickButton('使用工作频道询问当前情况');
    const channel = await evaluate(`document.querySelector('select[aria-label="发言范围"]').value`);
    assert.equal(channel, 'broadcast', 'Historical turn result must not erase fresh suggestion metadata');
    assert.equal(report.posted_turns.length, posts);
    assert.deepEqual(await api(`/api/campaigns/${historicalCampaign}`), before);
    report.historical_recovery = {campaign_id: historicalCampaign, turn_id: historicalTurn, channel, extra_posts: 0, campaign_unchanged: true};
    await screenshot('historical-recovery');
    step('historical turn reload preserved current suggestion routing without submitting', report.historical_recovery);
  }
  assert.deepEqual(report.errors, []);
  report.completed_at = new Date().toISOString();
  report.result = 'passed';
}
async function realPrivateEvidence(first, original) {
  assert.equal(original.mock, false, 'Refuse to claim a mock adventure is real');
  const doctor = original.npcs.find(npc => npc.id === 'doctor');
  assert.ok(doctor, 'Public roster must contain the doctor');
  assert.ok(!original.evidence.some(evidence => evidence.id === 'E05'), 'E05 must be newly obtained during this test');
  await evaluate(`(() => {const el=document.querySelector('select[aria-label="发言范围"]');el.value='private';el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await waitFor(`document.querySelector('.recipients input[value="doctor"]')`, 'doctor private recipient');
  await clickExpression(`document.querySelector('.recipients input[value="doctor"]')`, `${doctor.name} recipient checkbox`);
  assert.equal(await evaluate(`document.querySelector('.recipients input[value="doctor"]').checked`), true);
  await fill('textarea[aria-label="你的行动"]', realCommand);
  await clickExpression(`document.querySelector('.aside-details > summary')`, 'expand the DM private aside');
  assert.equal(await evaluate(`document.querySelector('.aside-details').open`), true);
  await fill('textarea[aria-label="给 DM 的私下说明"]', privateCanary);
  assert.equal(await evaluate(`document.querySelector('textarea[aria-label="给 DM 的私下说明"]').value`), privateCanary);
  await screenshot('private-input');
  step('private doctor request prepared in UI', { recipient_name: doctor.name, channel: 'private', text: realCommand });
  report.before_submission_health = await api('/api/health');
  assert.equal(report.before_submission_health.model_available, true, 'Runner became unavailable before submission');
  assert.equal(report.before_submission_health.model_busy, false, 'Runner became busy; no real request was sent');
  assert.equal(report.before_submission_health.mock, false);
  await clickButton('采取行动');
  const terminal = await waitTurn(first.campaignId, null, 180000);
  step('real private turn reached terminal', report.real_terminal);
  const after = await api(`/api/campaigns/${first.campaignId}`);
  const transcriptResponse = await fetch(`${base}/api/campaigns/${first.campaignId}/transcript`);
  assert.equal(transcriptResponse.ok, true, 'Public transcript must be downloadable for aside isolation audit');
  const publicTranscript = await transcriptResponse.text();
  fs.writeFileSync(path.join(output, `${artifactPrefix}-transcript.md`), publicTranscript);
  const visiblePublic = await evaluate(`({ log: document.querySelector('[role="log"]').innerText, npc: Array.from(document.querySelectorAll('.npc-entry .entry-text')).map(el=>el.innerText).join('\\n') })`);
  report.private_aside_audit = {
    canary: privateCanary,
    post_private_aside_exact: report.posted_turns[0]?.private_aside === privateCanary,
    terminal_contains_canary: JSON.stringify(terminal).includes(privateCanary),
    player_view_contains_canary: JSON.stringify(after).includes(privateCanary),
    public_transcript_contains_canary: publicTranscript.includes(privateCanary),
    visible_player_log_contains_canary: visiblePublic.log.includes(privateCanary),
    visible_npc_content_contains_canary: visiblePublic.npc.includes(privateCanary),
  };
  assert.equal(report.private_aside_audit.post_private_aside_exact, true, 'The canary must be in the actual outgoing POST.private_aside');
  for (const [surface, leaked] of Object.entries(report.private_aside_audit)) {
    if (surface.endsWith('_contains_canary')) assert.equal(leaked, false, `Private aside leaked into ${surface}`);
  }
  step('actual DM aside retained in POST and absent from public outputs', report.private_aside_audit);
  // A failed, cancelled, needs-input or degraded response is evidence of failure, not a cue to fake a card or retry.
  assert.equal(terminal.status, 'committed', `Real private turn ended as ${terminal.status}: ${terminal.public_reply}`);
  assert.ok(terminal.result, 'Committed real turn must include its result');
  assert.equal(terminal.result.degraded, false, 'Real private turn used degraded generation; retain failure report');
  assert.equal(after.mock, false);
  const evidence = after.evidence.find(item => item.id === 'E05');
  report.evidence_after = after.evidence.map(item => ({ id: item.id, title: item.title }));
  assert.ok(evidence, 'The real model turn did not grant E05; no synthetic card will be inserted');
  await waitFor(`Array.from(document.querySelectorAll('.evidence-card summary')).some(el=>el.textContent.trim()===${JSON.stringify(evidence.title)})`, 'actual E05 evidence card in player UI');
  await clickExpression(`Array.from(document.querySelectorAll('.evidence-card summary')).find(el=>el.textContent.trim()===${JSON.stringify(evidence.title)})`, 'open E05 card');
  const renderedCard = await evaluate(`(() => { const s=Array.from(document.querySelectorAll('.evidence-card summary')).find(el=>el.textContent.trim()===${JSON.stringify(evidence.title)});return {open:s.parentElement.open,text:s.parentElement.innerText};})()`);
  assert.equal(renderedCard.open, true);
  assert.ok(renderedCard.text.includes(evidence.text), 'The actual evidence text must be rendered, not just its title');
  report.e05 = { id: evidence.id, title: evidence.title, text: evidence.text, source: evidence.source, rendered: renderedCard };
  await screenshot('evidence');
  step('real E05 card acquired and opened', { evidence_id: evidence.id, title: evidence.title, tick: after.tick, revision: after.revision });
  await clickButton('战役与存档');
  await waitFor(`document.querySelector('dialog[open] .save-form input')`, 'real evidence save dialog');
  const saveName = '真实私聊验收_E05';
  await fill('dialog[open] .save-form input', saveName);
  await clickButton('保存进度');
  await waitFor(`Array.from(document.querySelectorAll('.saved-row strong')).some(el=>el.textContent===${JSON.stringify(saveName)})`, 'real evidence named save');
  const saves = await api(`/api/campaigns/${first.campaignId}/saves`);
  const saved = saves.find(item => item.name === saveName);
  assert.ok(saved, 'Named save must exist in the real API');
  await screenshot('save-dialog');
  step('real evidence named save created', { save_id: saved.save_id, name: saved.name });
  await clickButton('从此处继续');
  const forked = await waitFor(`(() => {const s=JSON.parse(localStorage.getItem('local-trpg.session.v1')||'null');return s&&s.campaignId!==${JSON.stringify(first.campaignId)}&&!document.querySelector('dialog[open]')?s:false;})()`, 'new real evidence branch');
  report.campaigns_created.push(forked.campaignId);
  const branch = await api(`/api/campaigns/${forked.campaignId}`);
  assert.ok(branch.evidence.some(item => item.id === 'E05' && item.text === evidence.text), 'E05 must survive save/fork');
  assert.equal(branch.tick, after.tick);
  assert.equal(branch.player.location_id, after.player.location_id);
  assert.equal((await api(`/api/campaigns/${first.campaignId}`)).revision, after.revision, 'Fork must preserve the original timeline');
  await waitFor(`Array.from(document.querySelectorAll('.evidence-card summary')).some(el=>el.textContent.trim()===${JSON.stringify(evidence.title)})`, 'restored E05 card');
  await clickExpression(`Array.from(document.querySelectorAll('.evidence-card summary')).find(el=>el.textContent.trim()===${JSON.stringify(evidence.title)})`, 'open restored E05');
  await screenshot('restored-evidence');
  step('real evidence restored in a separate branch', { campaign_id: forked.campaignId, evidence_id: 'E05', tick: branch.tick });
  const postsBeforeReload = report.posted_turns.length;
  await send('Page.reload', { ignoreCache: true });
  await waitFor(`document.querySelector('textarea[aria-label="你的行动"]')`, 'real evidence campaign after reload');
  await delay(1600);
  assert.equal((await session()).campaignId, forked.campaignId);
  assert.equal(report.posted_turns.length, postsBeforeReload, 'Reload must not repost the private action');
  const request = report.posted_turns[0];
  assert.equal(report.posted_turns.length, 1, 'Real mode sends precisely one game action');
  assert.equal(request.mode, 'in_character');
  assert.equal(request.speech_channel, 'private');
  assert.deepEqual(request.recipient_ids, ['doctor']);
  assert.equal(request.text, realCommand);
  assert.equal(request.private_aside, privateCanary);
  report.private_aside_audit.reloaded_player_log_contains_canary = await evaluate(`document.querySelector('[role="log"]').innerText.includes(${JSON.stringify(privateCanary)})`);
  assert.equal(report.private_aside_audit.reloaded_player_log_contains_canary, false, 'Restored player log must not reveal the aside');
  assert.deepEqual(report.errors, [], 'Unexpected real-browser console errors');
  step('real private request audited; refresh caused no duplicate');
  report.completed_at = new Date().toISOString();
  report.result = 'passed';
}
async function main() {
  chromeProcess = spawn(chrome, ['--headless=new', '--disable-gpu', '--no-proxy-server', `--remote-debugging-port=${debugPort}`, '--remote-debugging-address=127.0.0.1', `--user-data-dir=${profile}`, '--window-size=1440,1000', '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-sync', '--disable-extensions', '--disable-component-update', '--disable-features=Translate,MediaRouter', 'about:blank'], { stdio: 'ignore', windowsHide: true });
  let endpoint;
  for (let i=0;i<100;i++) {
    try { const info = await (await fetch(`http://127.0.0.1:${debugPort}/json/version`)).json(); report.browser = info.Browser; endpoint = (await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json()).find(t=>t.type==='page'); if (endpoint) break; } catch {}
    await delay(200);
  }
  if (!endpoint) throw new Error('Chrome DevTools did not become ready');
  socket = new WebSocket(endpoint.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  socket.addEventListener('message', event => {
    const message=JSON.parse(String(event.data));
    if (message.id) { const promise=pending.get(message.id); if (promise) {pending.delete(message.id);message.error?promise.reject(new Error(message.error.message)):promise.resolve(message.result);} return; }
    if (message.method==='Runtime.exceptionThrown') report.errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
    if (message.method==='Runtime.consoleAPICalled' && ['error','warning'].includes(message.params.type)) (message.params.type==='error'?report.errors:report.warnings).push(message.params.args.map(a=>a.value||a.description||'').join(' '));
    if (message.method==='Log.entryAdded' && ['error','warning'].includes(message.params.entry.level)) (message.params.entry.level==='error'?report.errors:report.warnings).push(message.params.entry.text + (message.params.entry.url ? ' ' + message.params.entry.url : ''));
    if (message.method==='Network.loadingFailed') report.failed_requests.push({error:message.params.errorText,canceled:message.params.canceled||false});
    if (message.method==='Network.requestWillBeSent') {
      const r=message.params.request;
      if(r.method==='POST' && /\/turns$/.test(new URL(r.url).pathname)) {const body=JSON.parse(r.postData||'{}');report.posted_turns.push({url:r.url,mode:body.mode,text:body.text,speech_channel:body.speech_channel,recipient_ids:body.recipient_ids,private_aside:body.private_aside,client_request_id:body.client_request_id});}
    }
  });
  for (const domain of ['Runtime','Page','Network','Log']) await send(domain+'.enable');
  await send('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:base});
  await waitFor(`Array.from(document.querySelectorAll('button')).some(b=>b.textContent.includes('开始冒险')&&!b.disabled)`, 'welcome page');
  await screenshot('welcome');
  step('welcome rendered');
  if (realMode) {
    const health = await api('/api/health');
    report.initial_health = health;
    assert.equal(health.model_available, true, 'Real mode requires the shared runner to be ready');
    assert.equal(health.model_busy, false, 'Shared runner is busy; do not overlap the parent playtest');
    assert.equal(health.mock, false, 'Real mode refuses a mock backend');
  }
  await clickButton('开始冒险');
  await waitFor(`document.querySelector('dialog[open] input[placeholder="船员们该怎么称呼你？"]')`, 'new campaign dialog');
  const playerName=(openingMode ? '开场按钮验收_' : realMode ? '真实私聊验收_' : '界面验收_')+new Date().toISOString().replace(/\D/g,'').slice(0,12);
  await fill('dialog[open] input[placeholder="船员们该怎么称呼你？"]',playerName);
  await clickButton('登上同路号');
  await waitFor(`document.querySelector('textarea[aria-label="你的行动"]') && !document.querySelector('dialog[open]')`, 'new campaign');
  const first = await session();
  report.campaigns_created.push(first.campaignId);
  const original = await api(`/api/campaigns/${first.campaignId}`);
  assert.equal(original.player.name,playerName);
  assert.equal(original.tick,0);
  await screenshot('desktop');
  step('campaign created through UI',{campaign_id:first.campaignId,tick:original.tick,npcs:original.npcs.length});
  assert.equal(original.npcs.length,5);
  if (openingMode) { await realOpening(first); return; }
  if (realMode) { await realPrivateEvidence(first, original); return; }
  await evaluate(`(() => {const el=document.querySelector('select[aria-label="输入方式"]');el.value='ooc';el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await fill('textarea[aria-label="你的行动"]','请说明公开规则。');
  await clickButton('询问主持人');
  const ooc = await waitTurn(first.campaignId,null);
  assert.equal(ooc.status,'completed_ooc');
  assert.equal((await api(`/api/campaigns/${first.campaignId}`)).tick,0);
  step('OOC completed without advancing time',{status:ooc.status});
  // Parent process may be warming or restarting the server. Wait without starting a model ourselves.
  for (let i=0;i<180;i++) {
    try {if ((await api('/api/health')).model_available) break;} catch {}
    if (i%10===0) console.log(JSON.stringify({waiting:'parent-started model warmup',seconds:i*2}));
    if(i===179)throw new Error('Parent-started model did not become ready');
    await delay(2000);
  }
  await evaluate(`(() => {const el=document.querySelector('select[aria-label="输入方式"]');el.value='in_character';el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await fill('textarea[aria-label="你的行动"]','前往动力舱');
  await clickButton('采取行动');
  const move = await waitTurn(first.campaignId,ooc.turn_id);
  assert.equal(move.status,'committed');
  const moved = await api(`/api/campaigns/${first.campaignId}`);
  assert.equal(moved.player.location_id,'engine');
  assert.equal(moved.tick,0);
  assert.equal(moved.revision,1);
  step('zero-cost MOVE completed',{status:move.status,tick:moved.tick,revision:moved.revision,location:moved.player.location});
  await clickButton('战役与存档');
  await waitFor(`document.querySelector('dialog[open] .save-form input')`, 'save dialog');
  const saveName='浏览器验收_动力舱';
  await fill('dialog[open] .save-form input',saveName);
  await clickButton('保存进度');
  await waitFor(`Array.from(document.querySelectorAll('.saved-row strong')).some(el=>el.textContent===${JSON.stringify(saveName)})`, 'saved entry');
  await screenshot('save-dialog');
  const saves=await api(`/api/campaigns/${first.campaignId}/saves`);
  assert.equal(saves.length,1);assert.equal(saves[0].name,saveName);
  step('named save created',{save_id:saves[0].save_id});
  await clickButton('从此处继续');
  const forked=await waitFor(`(() => {const s=JSON.parse(localStorage.getItem('local-trpg.session.v1')||'null');return s&&s.campaignId!==${JSON.stringify(first.campaignId)}&&!document.querySelector('dialog[open]')?s:false;})()`, 'new branch from save');
  report.campaigns_created.push(forked.campaignId);
  const branch=await api(`/api/campaigns/${forked.campaignId}`);
  assert.equal(branch.player.location_id,'engine');assert.equal(branch.tick,0);
  assert.equal((await api(`/api/campaigns/${first.campaignId}`)).revision,1);
  step('save fork restored as a separate campaign',{campaign_id:forked.campaignId});
  const postsBeforeReload=report.posted_turns.length;
  await send('Page.reload',{ignoreCache:true});
  await waitFor(`document.querySelector('textarea[aria-label="你的行动"]')`, 'restored campaign after page reload');
  await delay(1600);
  assert.equal((await session()).campaignId,forked.campaignId);
  assert.equal(report.posted_turns.length,postsBeforeReload);
  step('reload restored campaign without posting another action');
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true,screenWidth:390,screenHeight:844});
  await evaluate('window.scrollTo(0,0)');
  await delay(400);
  const layout=await evaluate(`({viewport:innerWidth,width:document.documentElement.scrollWidth,bodyWidth:document.body.scrollWidth,input:!!document.querySelector('textarea[aria-label="你的行动"]')})`);
  assert.ok(layout.width<=390,`Horizontal overflow: ${layout.width}`);
  await screenshot('mobile',true);
  await clickExpression(`document.querySelector('.mobile-panel-toggle button')`,'mobile character panel');
  assert.notEqual(await evaluate(`getComputedStyle(document.querySelector('.player-panel')).display`),'none');
  await screenshot('mobile-panel',true);
  step('390px mobile layout has no horizontal overflow',{...layout});
  await clickExpression(`document.querySelector('.mobile-panel-toggle button')`,'close mobile character panel');
  await clickButton('操作说明');
  await waitFor(`document.querySelector('dialog[open] .help-content')`,'onboarding help');
  await screenshot('mobile-help');
  await clickButton('明白了，回到航程');
  assert.equal(report.posted_turns.length,2);
  assert.ok(report.posted_turns.every(t=>t.mode==='ooc'||t.text==='前往动力舱'));
  step('onboarding help opened and dismissed');
  assert.deepEqual(report.errors.filter(message => !message.includes('/favicon.ico')), [], 'Unexpected browser console errors');
  report.completed_at=new Date().toISOString();
  report.result='passed';
}
main().catch(async error=>{report.result='failed';report.failure=error.stack;console.error(error.stack);process.exitCode=1;try{await screenshot('failure');report.failure_dom=await evaluate('document.body.innerText');}catch{}}).finally(async()=>{
  if(socket&&socket.readyState===WebSocket.OPEN){try{await send('Browser.close');}catch{} socket.close();}
  if(chromeProcess&&!chromeProcess.killed)chromeProcess.kill();
  await delay(600);
  try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:8,retryDelay:250});}catch(error){report.warnings.push('Temporary profile cleanup failed: '+error.message);}
  fs.writeFileSync(path.join(output,`${artifactPrefix}-report.json`),JSON.stringify(report,null,2)+'\n');
  if(realMode)fs.writeFileSync(path.join(output,openingMode?'browser-opening-latest-report.json':'browser-real-latest-report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({result:report.result,errors:report.errors,warnings:report.warnings,screenshots:report.screenshots,report:path.join(output,`${artifactPrefix}-report.json`)}));
});
