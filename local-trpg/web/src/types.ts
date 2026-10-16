export interface CharacterCard {
 id: string; name: string; role: string; hp: number; ac: number; skills: Record<string, number>;
 attacks: Attack[]; ability: Ability;
}
export interface Attack { id: string; name: string; bonus: number; dice?: number; damage_bonus?: number; fixed_damage?: number }
export interface Ability { id: string; name: string; uses: number | null; remaining?: number | null; description?: string; heal?: number }
export interface ModuleInfo { module_id: string; title: string; description: string; opening: string; ruleset?: string; characters?: CharacterCard[]; source_url?: string; adaptations?: string[] }
export interface Health { status: string; backend: string; model_available: boolean; model_busy: boolean; mock: boolean }
export interface CampaignInfo { campaign_id: string; module_id?: string; player_name: string; revision: number; tick: number; updated_at: string }
export interface SaveInfo { save_id: string; name: string; revision: number; tick: number }
export interface TableTalk { turn_id: string; question: string; answer: string; created: number; reply_source: 'model' | 'legacy' }
export interface PlayerView {
  campaign_id: string; revision: number; module_title: string;
  module_id?: string; ruleset?: string; skill_names?: Record<string, string>;
  player: { id: string; name: string; role?: string; skills?: Record<string, number>; location_id: string; location: string; health: string; inventory: Array<{ id: string; name: string }>; hp?: number; max_hp?: number; ac?: number; attacks?: Attack[]; ability?: Ability; ammo?: number };
  scene?: { description: string; visible_objects: string[]; exits: Array<{ id: string; name: string }>; investigations: Array<{ id: string; title: string }> };
  party?: Array<{ id: string; name: string; role: string; hp: number; max_hp: number; ac: number; ability: Ability }>;
  combat?: null | { round: number; enemies: Array<{ id: string; name: string; condition: string }> };
  tick: number; phase: string; clocks: Array<{ label: string; value: number; limit: number }>;
  locations: Array<{ id: string; name: string }>;
  npcs: Array<{ id: string; name: string; role: string; location_id: string | null; present: boolean }>;
  evidence: Array<{ id: string; title: string; text: string; source: string }>;
  suggestions: string[]; transcript: Array<{ turn_id: string; speaker: string; speaker_id?: string; text: string; tick: number }>;
  suggestion_commands?: Record<string, Omit<CommandFields, 'private_aside'>>;
  ending_summary?: string[]; table_talk?: TableTalk[];
  outcome: null | { crew_saved: number; cells_saved: number; ship_saved: boolean; cause_verified: boolean; captain_action_verified: boolean } | { kind: 'adventure'; status: string; summary: string; wyrmling_returned: boolean; care_accepted: boolean };
  active_turn_id: string | null; mock: boolean;
}
export interface CommandFields { mode: 'in_character' | 'ooc'; text: string; speech_channel: 'local' | 'private' | 'broadcast'; recipient_ids: string[]; private_aside: string }
export interface PlayerCommand extends CommandFields { client_request_id: string; expected_revision: number; reply_to_turn_id: string | null }
export interface PublicCheck { label?: string; skill?: string; dice?: number[]; total?: number; outcome?: string; cost?: string; difficulty?: number }
export interface ProvisionalReplies {
  campaign_id: string; turn_id: string; epoch: number; base_revision: number;
  dialogue: Array<{ event_id: string; speaker: string; speaker_id: string; text: string }>;
}
export interface TurnStatus {
  mode?: 'in_character' | 'ooc';
  reply_source?: 'model' | 'legacy' | 'system'; table_talk?: TableTalk[];
  campaign_id?: string; epoch?: number; base_revision?: number; provisional?: ProvisionalReplies | null;
  turn_id: string; status: 'queued' | 'running' | 'committed' | 'completed_ooc' | 'needs_input' | 'failed' | 'cancelled' | 'recoverable';
  stage: string; is_terminal: boolean; public_reply: string; revision: number;
  result: null | { narration: string; dialogue: Array<{ speaker: string; speaker_id?: string; text: string }>; checks: PublicCheck[]; degraded: boolean; view: PlayerView; rule_log?: string[] };
  cancel_requested: boolean;
}
