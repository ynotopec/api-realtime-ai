const mode = import.meta.env.MODE ?? 'development';
const env = import.meta.env;

const devWsBase = env.VITE_REALTIME_API_BASE ?? 'ws://localhost:8080';
const prodWsBase = env.VITE_REALTIME_API_BASE_PROD ?? env.VITE_REALTIME_API_BASE ?? devWsBase;

export const realtimeConfig = {
  wsBaseUrl: mode === 'production' ? prodWsBase : devWsBase,
  apiKey: env.VITE_REALTIME_API_KEY ?? env.VITE_OPENAI_API_KEY,
  channel: env.VITE_REALTIME_CHANNEL
};

export function buildRealtimeUrl(base: string, channel?: string) {
  const normalized = base.endsWith('/') ? base.slice(0, -1) : base;
  const url = `${normalized}/v1/realtime`;
  if (!channel) return url;
  const hasQuery = url.includes('?');
  const sep = hasQuery ? '&' : '?';
  return `${url}${sep}channel=${encodeURIComponent(channel)}`;
}
