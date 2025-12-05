import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const baseUrl = __ENV.BASE_URL || 'http://localhost:8080';
const wsUrl = __ENV.WS_URL || `${baseUrl.replace(/^http/, 'ws')}/v1/realtime`;
const feedbackUrl = __ENV.FEEDBACK_URL || `${baseUrl}/feedback`;
const apiToken = __ENV.API_TOKEN || '';

const connectionErrors = new Rate('chat_stream_errors');
const chatRoundTrip = new Trend('chat_roundtrip_ms');
const feedbackLatency = new Trend('feedback_latency_ms');
const feedbackErrors = new Counter('feedback_errors');

const chatVUs = parseInt(__ENV.CHAT_VUS || '5', 10);
const chatDuration = __ENV.CHAT_DURATION || '2m';
const feedbackRate = parseInt(__ENV.FEEDBACK_RATE || '20', 10);
const feedbackDuration = __ENV.FEEDBACK_DURATION || chatDuration;
const gracefulStop = __ENV.GRACEFUL_STOP || '30s';
const chatTimeoutMs = parseInt(__ENV.CHAT_TIMEOUT_MS || '12000', 10);

export const options = {
  scenarios: {
    chat_streams: {
      executor: 'constant-vus',
      vus: chatVUs,
      duration: chatDuration,
      gracefulStop,
      exec: 'chat_streams',
    },
    feedback_posts: {
      executor: 'constant-arrival-rate',
      rate: feedbackRate,
      timeUnit: '1m',
      duration: feedbackDuration,
      preAllocatedVUs: Math.max(5, feedbackRate),
      maxVUs: Math.max(10, feedbackRate * 2),
      gracefulStop,
      exec: 'feedback_posts',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.02'],
    checks: ['rate>0.95'],
    chat_roundtrip_ms: ['p(95)<8000'],
    feedback_latency_ms: ['p(95)<2000'],
    chat_stream_errors: ['rate<0.05'],
  },
  summaryTrendStats: ['avg', 'min', 'max', 'p(90)', 'p(95)'],
};

function authHeaders() {
  return apiToken ? { Authorization: `Bearer ${apiToken}` } : {};
}

function buildMessage(text) {
  return {
    type: 'conversation.item.create',
    item: {
      type: 'message',
      role: 'user',
      content: [{ type: 'input_text', text }],
    },
  };
}

function requestResponse() {
  return { type: 'response.create', response: {} };
}

export function chat_streams() {
  const params = { headers: authHeaders() };
  const start = Date.now();
  let receivedFirstDelta = false;

  const res = ws.connect(wsUrl, params, (socket) => {
    socket.on('open', () => {
      socket.send(
        JSON.stringify({
          type: 'session.update',
          session: { voice: 'shimmer', modalities: ['audio', 'text'] },
        }),
      );
      socket.send(JSON.stringify(buildMessage('Say hello and acknowledge this is a load test.')));
      socket.send(JSON.stringify(requestResponse()));
    });

    socket.on('message', (data) => {
      try {
        const payload = JSON.parse(data);
        if (!receivedFirstDelta && payload.type === 'response.output_text.delta') {
          receivedFirstDelta = true;
          chatRoundTrip.add(Date.now() - start);
          socket.close();
        }
        if (payload.type === 'response.error') {
          connectionErrors.add(1);
        }
      } catch (err) {
        connectionErrors.add(1);
      }
    });

    socket.on('error', () => {
      connectionErrors.add(1);
    });

    socket.setTimeout(() => {
      connectionErrors.add(1);
      socket.close();
    }, chatTimeoutMs);
  });

  check(res, {
    'ws connected': (r) => r && r.status === 101,
  });
}

export function feedback_posts() {
  const payload = {
    session_id: `session-${__ITER}`,
    rating: 5,
    comment: 'Automated load test feedback',
  };

  const res = http.post(feedbackUrl, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    timeout: chatTimeoutMs,
  });

  feedbackLatency.add(res.timings.duration);
  const ok = check(res, {
    'feedback accepted': (r) => r.status >= 200 && r.status < 300,
  });

  if (!ok) {
    feedbackErrors.add(1);
  }

  sleep(1);
}

export function handleSummary(data) {
  return {
    'load-tests/k6-summary.json': JSON.stringify(data, null, 2),
  };
}
