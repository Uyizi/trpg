import type { Health } from './types';

export function ModelAvailability({ health, warming, onWarmup }: { health: Health | null; warming: boolean; onWarmup: () => void }) {
  const loading = warming || health?.status === 'loading';
  const ready = Boolean(health?.model_available);
  const label = health?.mock ? '测试模式' : ready ? '本地主持人在线' : loading ? '主持人准备中' : health ? '主持人尚未启动' : '等待本地连接';
  return <div className="model-availability">
    <span className={`connection ${ready ? 'ready' : ''}`} role="status"><span className="status-dot" />{label}</span>
    {health && !health.mock && !ready && !loading && <button className="topbar-button warmup-button" onClick={onWarmup}>启动主持人</button>}
    {loading && <span className="warmup-progress">首次准备可能需要几分钟</span>}
  </div>;
}
