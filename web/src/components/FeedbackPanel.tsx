import { Feedback } from '../types';

interface Props {
  feedback: Feedback;
  onFeedback: (feedback: Feedback) => void;
}

export function FeedbackPanel({ feedback, onFeedback }: Props) {
  return (
    <div className="feedback-buttons">
      <button
        type="button"
        aria-pressed={feedback === 'positive'}
        onClick={() => onFeedback('positive')}
      >
        👍 Helpful
      </button>
      <button
        type="button"
        aria-pressed={feedback === 'negative'}
        onClick={() => onFeedback('negative')}
      >
        👎 Not quite
      </button>
    </div>
  );
}
