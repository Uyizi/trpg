import { ApiError, errorMessage, request } from './api';
import type { CampaignInfo, CommandFields, Health, ModuleInfo, PlayerCommand, PlayerView, SaveInfo, TurnStatus } from './types';

export interface PendingRequest { campaignId: string; command: PlayerCommand }
interface SavedSession { campaignId: string; turnId: string | null; pending: PendingRequest | null }
export interface ClientState {
  view: PlayerView | null; health: Health | null; modules: ModuleInfo[]; campaigns: CampaignInfo[];
  turnId: string | null; turn: TurnStatus | null; pending: PendingRequest | null; acceptedRequestId: string | null;
  loading: boolean; submitting: boolean; error: string | null; pollError: string | null; notice: string | null;
  storageAvailable: boolean; warming: boolean;
}
const SESSION_KEY = 'local-trpg.session.v1';

export function actionBlocked(state: ClientState): boolean {
  return state.loading || state.submitting || Boolean(state.pending) ||
    Boolean(state.turnId && !state.turn) || Boolean(state.turn && (!state.turn.is_terminal || state.turn.status === 'recoverable'));
}

export class GameClient {
  private state: ClientState = {
    view: null, health: null, modules: [], campaigns: [], turnId: null, turn: null, pending: null, acceptedRequestId: null,
    loading: true, submitting: false, error: null, pollError: null, notice: null, storageAvailable: true, warming: false,
  };
  private listeners = new Set<() => void>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private pollEpoch = 0;
  private selectionEpoch = 0;
  private healthTimer: ReturnType<typeof setTimeout> | null = null;
  private healthEpoch = 0;
  getSnapshot = () => this.state;
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  private update(values: Partial<ClientState>) {
    this.state = { ...this.state, ...values };
    this.listeners.forEach(listener => listener());
  }
  clearMessage = () => this.update({ error: null, notice: null });
  private readSession(): SavedSession | null {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (!raw) return null;
      const value = JSON.parse(raw) as SavedSession;
      if (typeof value.campaignId !== 'string' || !value.campaignId) return null;
      if (value.pending && (value.pending.campaignId !== value.campaignId || typeof value.pending.command?.client_request_id !== 'string' || typeof value.pending.command?.text !== 'string')) return null;
      return { campaignId: value.campaignId, turnId: typeof value.turnId === 'string' ? value.turnId : null, pending: value.pending ?? null };
    } catch { return null; }
  }
  private persist() {
    if (!this.state.view) return;
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify({ campaignId: this.state.view.campaign_id, turnId: this.state.turnId, pending: this.state.pending }));
    } catch { this.update({ storageAvailable: false }); }
  }
  async start() {
    const saved = this.readSession();
    const results = await Promise.allSettled([
      request<Health>('/api/health'), request<ModuleInfo[]>('/api/modules'), request<CampaignInfo[]>('/api/campaigns'),
    ]);
    const [health, modules, campaigns] = results;
    this.update({
      health: health.status === 'fulfilled' ? health.value : null,
      modules: modules.status === 'fulfilled' ? modules.value : [],
      campaigns: campaigns.status === 'fulfilled' ? campaigns.value : [], loading: false,
      error: modules.status === 'rejected' ? errorMessage(modules.reason) : null,
    });
    if (health.status === 'fulfilled') this.acceptHealth(health.value);
    if (saved) await this.openCampaign(saved.campaignId, saved);
  }
  async refreshLibrary() {
    const [campaigns, health] = await Promise.allSettled([request<CampaignInfo[]>('/api/campaigns'), request<Health>('/api/health')]);
    if (campaigns.status === 'fulfilled') this.update({ campaigns: campaigns.value });
    else this.update({ error: errorMessage(campaigns.reason) });
    if (health.status === 'fulfilled') this.acceptHealth(health.value);
  }
  async openCampaign(id: string, restore?: SavedSession): Promise<boolean> {
    const selection = ++this.selectionEpoch;
    this.stopPolling();
    this.update({ loading: true, turn: null, error: null, pollError: null });
    try {
      const view = await request<PlayerView>(`/api/campaigns/${encodeURIComponent(id)}`);
      if (selection !== this.selectionEpoch) return false;
      const turnId = view.active_turn_id || restore?.turnId || null;
      this.update({ view, turnId, turn: null, pending: restore?.pending ?? null, submitting: false, loading: false, notice: null });
      this.persist();
      if (turnId) this.beginPolling();
      return true;
    } catch (error) {
      if (selection === this.selectionEpoch) this.update({ loading: false, error: errorMessage(error) });
      return false;
    }
  }
  private adopt(view: PlayerView) {
    ++this.selectionEpoch; this.stopPolling();
    this.update({ view, turnId: view.active_turn_id, turn: null, pending: null, error: null, pollError: null, notice: null, loading: false });
    this.persist();
    if (view.active_turn_id) this.beginPolling();
    void this.refreshLibrary();
  }
  async createCampaign(moduleId: string, name: string, characterId?: string): Promise<boolean> {
    this.update({ loading: true, error: null });
    try {
      const data = await request<{ campaign_id: string; view: PlayerView }>('/api/campaigns', { module_id: moduleId, player_setup: { name: name.trim(), ...(characterId ? { character_id: characterId } : {}) } });
      this.adopt(data.view); return true;
    } catch (error) { this.update({ loading: false, error: errorMessage(error) }); return false; }
  }
  async submit(fields: CommandFields): Promise<boolean> {
    const { view, turn } = this.state;
    if (!view || actionBlocked(this.state) || !fields.text.trim()) return false;
    if (view.outcome && fields.mode !== 'ooc') return false;
    const command: PlayerCommand = {
      ...fields, text: fields.text.trim(), recipient_ids: [...fields.recipient_ids],
      client_request_id: crypto.randomUUID(), expected_revision: view.revision,
      reply_to_turn_id: turn?.status === 'needs_input' ? turn.turn_id : null,
    };
    this.update({ pending: { campaignId: view.campaign_id, command }, error: null, notice: null });
    this.persist();
    return this.sendPending();
  }
  retrySubmission = async (): Promise<boolean> => {
    if (!this.state.pending || this.state.submitting) return false;
    return this.sendPending();
  };
  private async sendPending(): Promise<boolean> {
    const pending = this.state.pending;
    if (!pending) return false;
    const selection = this.selectionEpoch;
    this.update({ submitting: true, error: null });
    try {
      const receipt = await request<{ turn_id: string; status: string }>(`/api/campaigns/${encodeURIComponent(pending.campaignId)}/turns`, pending.command);
      if (selection !== this.selectionEpoch) return false;
      this.update({ pending: null, submitting: false, turnId: receipt.turn_id, turn: null, acceptedRequestId: pending.command.client_request_id });
      this.persist(); this.beginPolling(); return true;
    } catch (error) {
      if (selection !== this.selectionEpoch) return false;
      // A timeout or 5xx can happen after acceptance. Preserve both ID and original body.
      const rejected = error instanceof ApiError && error.status >= 400 && error.status < 500;
      this.update({ submitting: false, error: errorMessage(error), pending: rejected ? null : pending });
      this.persist();
      if (error instanceof ApiError && error.status === 409) {
        const id = pending.campaignId;
        try {
          const view = await request<PlayerView>(`/api/campaigns/${encodeURIComponent(id)}`);
          if (selection !== this.selectionEpoch) return false;
          this.update({ view, turnId: view.active_turn_id, turn: null }); this.persist();
          if (view.active_turn_id) this.beginPolling();
        } catch { /* Keep the explicit original conflict and editable draft. */ }
      }
      return false;
    }
  }
  private stopPolling() {
    ++this.pollEpoch;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }
  private acceptStatus(turn: TurnStatus) {
    const current = this.state.view;
    if (!current || turn.turn_id !== this.state.turnId || (turn.campaign_id && turn.campaign_id !== current.campaign_id)) return true;
    if (this.state.turn?.turn_id === turn.turn_id && (turn.epoch ?? 0) < (this.state.turn.epoch ?? 0)) return true;
    if (current?.active_turn_id && current.active_turn_id !== turn.turn_id) {
      // An old retry receipt must not replace a different authoritative active turn.
      this.update({ turnId: current.active_turn_id, turn: null, pollError: null });
      this.persist(); this.beginPolling(); return false;
    }
    const preview = turn.provisional;
    const validPreview = preview && !turn.is_terminal && turn.status === 'running' && !turn.cancel_requested &&
      preview.campaign_id === current.campaign_id && preview.turn_id === turn.turn_id && preview.epoch === turn.epoch &&
      preview.base_revision === turn.base_revision && preview.base_revision === current.revision;
    turn = { ...turn, provisional: validPreview ? preview : null };
    const result = turn.result?.view?.campaign_id === current.campaign_id ? turn.result.view : null;
    // Equal revisions can still contain freshly projected metadata from a campaign read.
    const view = result && (!current || result.revision > current.revision) ? result : current;
    this.update({
      turn, pollError: null,
      view: view ? { ...view, table_talk: turn.table_talk ?? view.table_talk, active_turn_id: turn.is_terminal && turn.status !== 'recoverable' ? null : turn.turn_id } : null,
    });
    this.persist();
    return true;
  }
  private beginPolling() {
    this.stopPolling();
    const epoch = this.pollEpoch;
    const campaignId = this.state.view?.campaign_id;
    const turnId = this.state.turnId;
    if (!campaignId || !turnId) return;
    const poll = async () => {
      try {
        const turn = await request<TurnStatus>(`/api/campaigns/${encodeURIComponent(campaignId)}/turns/${encodeURIComponent(turnId)}`);
        if (epoch !== this.pollEpoch) return;
        if (!this.acceptStatus(turn)) return;
        if (this.state.turn?.is_terminal) return;
      } catch (error) {
        if (epoch !== this.pollEpoch) return;
        this.update({ pollError: errorMessage(error), turn: this.state.turn ? { ...this.state.turn, provisional: null } : null });
        if (error instanceof ApiError && error.status === 404) {
          try {
            const view = await request<PlayerView>(`/api/campaigns/${encodeURIComponent(campaignId)}`);
            if (epoch !== this.pollEpoch) return;
            if (view.active_turn_id !== turnId) {
              this.update({ view, turnId: view.active_turn_id, turn: null, pollError: null, notice: '已重新读取战役进度，请核对最新记录后继续。' });
              this.persist();
              if (view.active_turn_id) this.beginPolling();
            }
          } catch { /* Keep the original error if the campaign cannot be verified. */ }
          return;
        }
        if (error instanceof ApiError && [403, 422].includes(error.status)) return;
      }
      if (epoch === this.pollEpoch) this.timer = setTimeout(poll, 1100);
    };
    void poll();
  }
  async refreshCampaign(): Promise<void> {
    if (!this.state.view || this.state.submitting) return;
    const { campaign_id: id } = this.state.view;
    await this.openCampaign(id, { campaignId: id, turnId: this.state.turnId, pending: this.state.pending });
  }
  async cancelTurn(): Promise<void> { await this.controlTurn('cancel'); }
  async resumeTurn(): Promise<void> {
    if (this.state.turn?.status === 'recoverable') await this.controlTurn('resume');
  }
  private async controlTurn(action: 'cancel' | 'resume') {
    const { view, turnId, submitting } = this.state;
    if (!view || !turnId || submitting) return;
    const selection = this.selectionEpoch;
    this.stopPolling(); this.update({ submitting: true, error: null, turn: this.state.turn ? { ...this.state.turn, provisional: null } : null });
    try {
      const turn = await request<TurnStatus>(`/api/campaigns/${encodeURIComponent(view.campaign_id)}/turns/${encodeURIComponent(turnId)}/${action}`, {});
      if (selection !== this.selectionEpoch || this.state.turnId !== turnId) return;
      if (this.acceptStatus(turn) && !turn.is_terminal) this.beginPolling();
    } catch (error) { if (selection === this.selectionEpoch) { this.update({ error: errorMessage(error) }); this.beginPolling(); } }
    finally { if (selection === this.selectionEpoch) this.update({ submitting: false }); }
  }
  async loadSaves(): Promise<SaveInfo[]> {
    if (!this.state.view) return [];
    return request<SaveInfo[]>(`/api/campaigns/${encodeURIComponent(this.state.view.campaign_id)}/saves`);
  }
  async save(name: string): Promise<boolean> {
    if (!this.state.view) return false;
    try {
      const saved = await request<{ save_id: string; revision: number; during_turn: boolean }>(`/api/campaigns/${encodeURIComponent(this.state.view.campaign_id)}/saves`, { name: name.trim() });
      this.update({ notice: saved.during_turn ? '已保存上一已完成回合；正在进行的行动尚未写入这个存档。' : '已保存这一刻，可以随时从这里继续。', error: null });
      return true;
    } catch (error) { this.update({ error: errorMessage(error) }); return false; }
  }
  async fork(saveId: string): Promise<boolean> {
    this.update({ loading: true, error: null });
    try {
      const data = await request<{ campaign_id: string; view: PlayerView }>(`/api/saves/${encodeURIComponent(saveId)}/fork`, {});
      this.adopt(data.view); return true;
    } catch (error) { this.update({ loading: false, error: errorMessage(error) }); return false; }
  }
  async importBackup(data: unknown): Promise<boolean> {
    this.update({ loading: true, error: null });
    try {
      const imported = await request<{ campaign_id: string; view: PlayerView }>('/api/import', data);
      this.adopt(imported.view); return true;
    } catch (error) { this.update({ loading: false, error: errorMessage(error) }); return false; }
  }
  private stopHealthPolling() {
    ++this.healthEpoch;
    if (this.healthTimer !== null) clearTimeout(this.healthTimer);
    this.healthTimer = null;
  }
  private acceptHealth(health: Health) {
    this.update({ health });
    this.stopHealthPolling();
    if (health.status !== 'loading') return;
    const epoch = this.healthEpoch;
    const poll = async () => {
      try {
        const next = await request<Health>('/api/health');
        if (epoch !== this.healthEpoch) return;
        this.update({ health: next });
        if (next.status !== 'loading') return;
      } catch { /* Retry read-only health checks; never start a model automatically. */ }
      if (epoch === this.healthEpoch) this.healthTimer = setTimeout(poll, 2500);
    };
    this.healthTimer = setTimeout(poll, 2500);
  }
  async warmupModel(): Promise<boolean> {
    if (this.state.warming || this.state.health?.status === 'loading' || this.state.health?.model_available) return false;
    this.update({ warming: true, error: null });
    try {
      await request<unknown>('/api/model/warmup', {});
      this.acceptHealth(await request<Health>('/api/health'));
      return true;
    } catch (error) { this.update({ error: errorMessage(error) }); return false; }
    finally { this.update({ warming: false }); }
  }
  dispose() { this.stopPolling(); this.stopHealthPolling(); }
}
