import { ChatMessage, Feedback } from '../types';

interface Props {
  messages: ChatMessage[];
  onReplay: (id: string) => void;
  onFeedback: (id: string, feedback: Feedback) => void;
}

export function HistoryPanel({ messages, onReplay, onFeedback }: Props) {
  return (
    <div className="panel">
      <h2>Message history</h2>
      <div className="history-list">
        {messages.map((message) => (
          <div className="history-item" key={message.id}>
            <div className="label">
              {message.role.toUpperCase()} — {new Date(message.createdAt).toLocaleTimeString()}
            </div>
            <p>
              {(message.tokens?.length ? `${message.content}${message.tokens.join('')}` : message.content).slice(0, 180) ||
                '(streaming...)'}
            </p>
            <div className="replay-controls">
              <button className="secondary" type="button" onClick={() => onReplay(message.id)}>
                Replay
              </button>
              {message.role === 'assistant' && (
                <>
                  <button className="secondary" type="button" onClick={() => onFeedback(message.id, 'positive')}>
                    👍
                  </button>
                  <button className="secondary" type="button" onClick={() => onFeedback(message.id, 'negative')}>
                    👎
                  </button>
                </>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
