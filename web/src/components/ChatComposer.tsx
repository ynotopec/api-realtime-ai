import { FormEvent, useState } from 'react';

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function ChatComposer({ onSend, disabled }: Props) {
  const [text, setText] = useState('');

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (!text.trim()) return;
    onSend(text.trim());
    setText('');
  };

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <textarea
        placeholder="Ask the assistant something..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
      />
      <div className="composer-actions">
        <span className="label">Markdown is supported in responses.</span>
        <button className="primary" type="submit" disabled={disabled}>
          Send
        </button>
      </div>
    </form>
  );
}
