import { API_BASE } from "./api";

export type Role = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  stats?: Record<string, string | number> | null;
  backendConnected?: boolean | null;
};

export type PersistedTurn = {
  prompt: string;
  response: string;
};

export type SessionHistoryMap = Record<string, PersistedTurn[]>;
export type AllSessionHistoryMap = Record<string, SessionHistoryMap>;

export type SessionHistoryPayload = {
  session_id: string;
  history: SessionHistoryMap;
};

export type ChatMessageRequest = {
  role: Role;
  content: string;
};

export type ChatMessageResponse = {
  text: string;
  stats: Record<string, string | number> | null;
  backendConnected: boolean;
};

export const CHAT_HISTORY_USER = "default";
export const INITIAL_ASSISTANT_TEXT =
  "Hi, I am your disaster assessment assistant. Select a question below or type your own question.";

export const chatKeys = {
  all: () => ["chat"] as const,
  sessions: () => ["chat", "sessions"] as const,
  session: (sessionId: string) => ["chat", "session", sessionId] as const,
};

export function createInitialAssistantMessage(): ChatMessage {
  return {
    id: Date.now().toString(),
    role: "assistant",
    content: INITIAL_ASSISTANT_TEXT,
    timestamp: new Date(),
  };
}

export function normalizeHistoryToMessages(history: SessionHistoryMap): ChatMessage[] {
  const turns = Object.values(history).flat();
  if (turns.length === 0) {
    return [createInitialAssistantMessage()];
  }

  const base = Date.now();
  const restored: ChatMessage[] = [];
  turns.forEach((turn, index) => {
    restored.push(
      {
        id: `${base}-${index * 2}`,
        role: "user",
        content: turn.prompt,
        timestamp: new Date(base + index * 2),
      },
      {
        id: `${base}-${index * 2 + 1}`,
        role: "assistant",
        content: turn.response,
        timestamp: new Date(base + index * 2 + 1),
      },
    );
  });

  return restored;
}

export async function fetchAllSessionHistory(): Promise<AllSessionHistoryMap> {
  const response = await fetch(`${API_BASE}/chat/history`);
  if (!response.ok) {
    throw new Error(`Failed to load chat history (HTTP ${response.status})`);
  }
  return (await response.json()) as AllSessionHistoryMap;
}

export async function fetchSessionHistory(sessionId: string): Promise<SessionHistoryMap> {
  const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
  if (!response.ok) {
    throw new Error(`Failed to load session history (HTTP ${response.status})`);
  }
  const payload = (await response.json()) as SessionHistoryPayload;
  return payload.history;
}

export async function createChatSession(): Promise<string> {
  const response = await fetch(`${API_BASE}/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Session creation failed (HTTP ${response.status})`);
  }

  const data = (await response.json()) as { session_id?: string };
  const sessionId = data.session_id?.trim();
  if (!sessionId) {
    throw new Error("Session creation returned an invalid session id");
  }

  return sessionId;
}

export async function persistSessionTurn(
  sessionId: string,
  prompt: string,
  responseText: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/history/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user: CHAT_HISTORY_USER,
      prompt,
      response: responseText,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to persist session history (HTTP ${response.status})`);
  }
}

export async function sendChatMessage(
  message: string,
  history: ChatMessageRequest[],
  sessionId: string,
): Promise<ChatMessageResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Chat-Session-Id": sessionId,
      },
      body: JSON.stringify({
        message,
        conversation_history: history,
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = (await response.json()) as {
      response?: string;
      message?: string;
      stats?: Record<string, string | number> | null;
    };

    return {
      text: data.response ?? data.message ?? "",
      stats: data.stats ?? null,
      backendConnected: true,
    };
  } catch {
    return {
      text: "I could not reach the chat API right now. Please try again in a moment.",
      stats: null,
      backendConnected: false,
    };
  }
}
