import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Feedback, ChatMessage } from '../types';
import { FeedbackPanel } from './FeedbackPanel';

interface Props {
  messages: ChatMessage[];
  onFeedback: (id: string, feedback: Feedback) => void;
}

export function MessageList({ messages, onFeedback }: Props) {
  return (
    <div className="chat-window">
      {messages.map((message) => {
        const text = message.tokens?.length ? `${message.content}${message.tokens.join('')}` : message.content;
        return (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="meta">
              <span>{message.role.toUpperCase()}</span>
              <span>{new Date(message.createdAt).toLocaleTimeString()}</span>
            </div>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{text || '*Waiting for response...*'}</ReactMarkdown>
            {message.role === 'assistant' && (
              <FeedbackPanel feedback={message.feedback} onFeedback={(fb) => onFeedback(message.id, fb)} />
            )}
          </div>
        );
      })}
    </div>
  );
}
