export type Role = 'user' | 'assistant' | 'system';
export type Feedback = 'positive' | 'negative' | null;

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  tokens?: string[];
  feedback: Feedback;
  streaming?: boolean;
}

export interface ClientConfig {
  wsBaseUrl: string;
  apiKey?: string;
  channel?: string;
}
