import { buildRealtimeUrl } from '../config';
import type { ChatMessage, ClientConfig } from '../types';

interface Handlers {
  onMessage?: (message: ChatMessage) => void;
  onPatch?: (id: string, delta: Partial<ChatMessage>) => void;
  onError?: (error: string) => void;
  onStatus?: (connected: boolean) => void;
}

interface CreateResponseOptions {
  itemId: string;
  systemPrompt?: string;
}

export class RealtimeClient {
  private ws?: WebSocket;
  private readonly handlers: Handlers;
  private readonly config: ClientConfig;
  private readonly knownMessages = new Set<string>();

  constructor(config: ClientConfig, handlers: Handlers) {
    this.config = config;
    this.handlers = handlers;
  }

  connect() {
    const url = buildRealtimeUrl(this.config.wsBaseUrl, this.config.channel);
    this.ws = new WebSocket(url, this.buildProtocols());
    this.ws.onopen = () => {
      this.handlers.onStatus?.(true);
      this.send({ type: 'session.update', session: { modalities: ['text'], instructions: 'Render concise answers.' } });
    };
    this.ws.onclose = () => this.handlers.onStatus?.(false);
    this.ws.onerror = () => this.handlers.onError?.('WebSocket error');
    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string);
        this.routeIncoming(payload);
      } catch (err) {
        this.handlers.onError?.(`Failed to parse message: ${String(err)}`);
      }
    };
  }

  disconnect() {
    this.ws?.close();
  }

  sendUserMessage(content: string) {
    const itemId = crypto.randomUUID();
    this.knownMessages.add(itemId);
    const message: ChatMessage = {
      id: itemId,
      role: 'user',
      content,
      createdAt: Date.now(),
      feedback: null
    };
    this.handlers.onMessage?.(message);
    this.send({
      type: 'conversation.item.create',
      item: {
        id: itemId,
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: content }]
      }
    });
    this.createResponse({ itemId });
  }

  replayResponse(messageId: string) {
    this.createResponse({ itemId: messageId });
  }

  sendFeedback(messageId: string, sentiment: 'positive' | 'negative') {
    this.handlers.onPatch?.(messageId, { feedback: sentiment });
    this.send({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: `Feedback for ${messageId}: ${sentiment}`
          }
        ]
      }
    });
  }

  private createResponse(options: CreateResponseOptions) {
    this.send({
      type: 'response.create',
      response: {
        conversation_item_id: options.itemId,
        instructions: options.systemPrompt,
        metadata: { channel: this.config.channel ?? 'default' }
      }
    });
  }

  private routeIncoming(payload: any) {
    const { type } = payload;
    if (!type) return;
    if (type === 'conversation.item.created') {
      const item = payload.item;
      if (item?.type === 'message') {
        const message: ChatMessage = {
          id: item.id ?? crypto.randomUUID(),
          role: item.role ?? 'assistant',
          content: this.stringifyContent(item.content) ?? '',
          createdAt: Date.now(),
          feedback: null,
          streaming: item.role === 'assistant'
        };
        this.knownMessages.add(message.id);
        this.handlers.onMessage?.(message);
      }
      return;
    }

    if (type === 'response.output_text.delta') {
      const id = payload.item_id ?? payload.response_id ?? payload.response?.id ?? crypto.randomUUID();
      const delta = payload.delta ?? payload.text ?? '';
      if (!this.knownMessages.has(id)) {
        this.knownMessages.add(id);
        this.handlers.onMessage?.({
          id,
          role: 'assistant',
          content: '',
          createdAt: Date.now(),
          feedback: null,
          streaming: true,
          tokens: []
        });
      }
      this.handlers.onPatch?.(id, { tokens: [delta], streaming: true });
      return;
    }

    if (type === 'response.completed') {
      const id = payload.item_id ?? payload.response?.id ?? payload.response_id;
      if (id) {
        this.handlers.onPatch?.(id, { streaming: false });
      }
      return;
    }

    if (type === 'error') {
      this.handlers.onError?.(payload.error?.message ?? 'Unknown error');
    }
  }

  private stringifyContent(content: any): string {
    if (!content) return '';
    if (Array.isArray(content)) {
      const textBlock = content.find((c) => c.type === 'output_text' || c.type === 'input_text');
      if (textBlock?.text) return textBlock.text;
    }
    if (typeof content === 'string') return content;
    return JSON.stringify(content);
  }

  private buildProtocols(): string[] {
    const protocols: string[] = [];
    if (this.config.apiKey) {
      protocols.push(`openai-insecure-api-key.${this.config.apiKey}`);
    }
    return protocols;
  }

  private send(payload: unknown) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.handlers.onError?.('WebSocket not connected');
      return;
    }
    this.ws.send(JSON.stringify(payload));
  }
}
