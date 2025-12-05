import { useEffect, useMemo, useState } from 'react';
import { realtimeConfig } from './config';
import { RealtimeClient } from './api/realtimeClient';
import { ChatComposer } from './components/ChatComposer';
import { HistoryPanel } from './components/HistoryPanel';
import { MessageList } from './components/MessageList';
import type { ChatMessage, Feedback } from './types';

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'system-intro',
      role: 'system',
      content: 'Connected to the realtime bridge. Responses render as markdown.',
      createdAt: Date.now(),
      feedback: null
    }
  ]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const client = useMemo(() => {
    const nextClient = new RealtimeClient(realtimeConfig, {
      onMessage: (message) => {
        setMessages((prev) => [...prev, { ...message, tokens: message.tokens ?? [] }]);
      },
      onPatch: (id, delta) => {
        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id !== id) return msg;
            const tokens = delta.tokens ? [...(msg.tokens ?? []), ...delta.tokens] : msg.tokens;
            return { ...msg, ...delta, tokens };
          })
        );
      },
      onError: (err) => setError(err),
      onStatus: (status) => setConnected(status)
    });
    return nextClient;
  }, []);

  useEffect(() => {
    client.connect();
    return () => client.disconnect();
  }, [client]);

  const handleSend = (text: string) => {
    setError(null);
    client.sendUserMessage(text);
  };

  const handleReplay = (id: string) => {
    setError(null);
    client.replayResponse(id);
  };

  const handleFeedback = (id: string, feedback: Feedback) => {
    setMessages((prev) => prev.map((msg) => (msg.id === id ? { ...msg, feedback } : msg)));
    client.sendFeedback(id, feedback === 'positive' ? 'positive' : 'negative');
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <HistoryPanel messages={messages} onReplay={handleReplay} onFeedback={handleFeedback} />
        <div className="panel">
          <h2>Feedback</h2>
          <p className="label">Use the controls below or the inline buttons to capture sentiment.</p>
        </div>
        <div className="panel">
          <h2>Connection</h2>
          <div className={`badge ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? 'Connected to realtime endpoint' : 'Disconnected'}
          </div>
          {realtimeConfig.channel && <div className="badge">Channel: {realtimeConfig.channel}</div>}
          {realtimeConfig.wsBaseUrl && <div className="label">Base URL: {realtimeConfig.wsBaseUrl}</div>}
        </div>
      </aside>
      <main className="main">
        <div className="header">
          <div>
            <h1>Realtime chat playground</h1>
            <div className="status">Markdown rendering, feedback capture, and replay controls.</div>
          </div>
          {error && <div className="badge disconnected">{error}</div>}
        </div>
        <MessageList messages={messages} onFeedback={handleFeedback} />
        <ChatComposer onSend={handleSend} disabled={!connected} />
      </main>
    </div>
  );
}
