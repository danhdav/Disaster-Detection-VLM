import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  chatKeys,
  createInitialAssistantMessage,
  normalizeHistoryToMessages,
  type AllSessionHistoryMap,
  type ChatMessage as Message,
} from "../lib/chatApi";
import {
  useChatSessionQuery,
  useChatSessionsQuery,
  useCreateChatSessionMutation,
  usePersistSessionTurnMutation,
  useSendChatMessageMutation,
} from "../hooks/useChatQueries";

interface ChatSessionEntry {
  id: string;
  createdAt: Date;
}

const SUGGESTIONS = [
  "How many structures were destroyed?",
  "How many buildings show major damage?",
  "How many buildings were undamaged?",
  "What steps are recommended next?",
];

const DAMAGE_COLORS: Record<string, string> = {
  destroyed: "#ef4444",
  major_damage: "#f97316",
  minor_damage: "#eab308",
  no_damage: "#22c55e",
  un_classified: "#6b7280",
  total_assessed: "#3b82f6",
};

function SuggestionChips({ onSelect }: { onSelect: (text: string) => void }) {
  return (
    <div className="surge-chat-suggestions">
      {SUGGESTIONS.map((suggestion) => (
        <button
          key={suggestion}
          className="surge-chat-chip"
          onClick={() => onSelect(suggestion)}
          type="button"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="surge-chat-typing-indicator">
      <span />
      <span />
      <span />
    </div>
  );
}

function StatCards({ stats }: { stats: Record<string, string | number> }) {
  return (
    <div className="surge-chat-stat-cards">
      {Object.entries(stats).map(([key, value]) => (
        <div
          key={key}
          className="surge-chat-stat-card"
          style={{ borderLeftColor: DAMAGE_COLORS[key] ?? "#6b7280" }}
        >
          <span className="surge-chat-stat-num" style={{ color: DAMAGE_COLORS[key] ?? "#e5e7eb" }}>
            {value}
          </span>
          <span className="surge-chat-stat-label">{key.replace(/_/g, " ")}</span>
        </div>
      ))}
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`surge-chat-message-row ${isUser ? "is-user" : "is-assistant"}`}>
      {!isUser ? (
        <div className="surge-chat-avatar surge-chat-avatar-assistant">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
          </svg>
        </div>
      ) : null}

      <div className="surge-chat-bubble-wrapper">
        <div className={`surge-chat-bubble ${isUser ? "is-user" : "is-assistant"}`}>
          {message.content}
          {message.stats ? <StatCards stats={message.stats} /> : null}
        </div>
        <div className={`surge-chat-meta ${isUser ? "is-user" : ""}`}>
          <span className="surge-chat-timestamp">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>
      </div>

      {isUser ? (
        <div className="surge-chat-avatar surge-chat-avatar-user">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        </div>
      ) : null}
    </div>
  );
}

export function ChatSidebar() {
  const queryClient = useQueryClient();
  const sessionsQuery = useChatSessionsQuery();
  const createSessionMutation = useCreateChatSessionMutation();
  const sendMessageMutation = useSendChatMessageMutation();
  const persistTurnMutation = usePersistSessionTurnMutation();

  const [draftMessages, setDraftMessages] = useState<Message[]>(() => [
    createInitialAssistantMessage(),
  ]);
  const [input, setInput] = useState("");
  const [isSessionsOpen, setIsSessionsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sessionQuery = useChatSessionQuery(sessionId);
  const isTyping = sendMessageMutation.isPending;

  const sessions = useMemo<ChatSessionEntry[]>(() => {
    const allHistory = sessionsQuery.data ?? {};
    return Object.keys(allHistory).map((id, index) => ({
      id,
      createdAt: new Date(Date.now() - index * 1000),
    }));
  }, [sessionsQuery.data]);

  const sessionMessagesFromHistory = useMemo(() => {
    if (!sessionId) return null;
    const sessionHistory = sessionsQuery.data?.[sessionId];
    if (!sessionHistory) return null;
    return normalizeHistoryToMessages(sessionHistory);
  }, [sessionId, sessionsQuery.data]);

  const messages = sessionId
    ? (sessionQuery.data ?? sessionMessagesFromHistory ?? [createInitialAssistantMessage()])
    : draftMessages;
  const showSuggestions = messages.length <= 1 && !isTyping;

  const handleCreateSession = async () => {
    setSessionId(null);
    setDraftMessages([createInitialAssistantMessage()]);
    setInput("");
  };

  const handleSelectSession = (selectedSessionId: string) => {
    const seededHistory = sessionsQuery.data?.[selectedSessionId];
    if (seededHistory) {
      queryClient.setQueryData(
        chatKeys.session(selectedSessionId),
        normalizeHistoryToMessages(seededHistory),
      );
    }
    setSessionId(selectedSessionId);
    setInput("");
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
  }, [input]);

  const sendMessage = async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || isTyping) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setInput("");
    let currentSessionId = sessionId;
    let workingMessages = sessionId ? messages : draftMessages;

    if (!currentSessionId) {
      const pendingDraftMessages = [...draftMessages, userMessage];
      setDraftMessages(pendingDraftMessages);
      workingMessages = pendingDraftMessages;

      try {
        const createdSessionId = await createSessionMutation.mutateAsync();
        currentSessionId = createdSessionId;
        setSessionId(createdSessionId);

        queryClient.setQueryData(chatKeys.sessions(), (previous?: AllSessionHistoryMap) => {
          const next = { ...previous };
          if (!next[createdSessionId]) next[createdSessionId] = {};
          return next;
        });
      } catch {
        setDraftMessages((previous) => [
          ...previous,
          {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "I could not create a chat session. Please try again in a moment.",
            timestamp: new Date(),
            stats: null,
            backendConnected: false,
          },
        ]);
        return;
      }
    }

    const resolvedSessionId = currentSessionId;
    if (!resolvedSessionId) return;

    const sessionMessages =
      queryClient.getQueryData<Message[]>(chatKeys.session(resolvedSessionId)) ?? workingMessages;
    const optimisticUserMessages = sessionMessages.some((message) => message.id === userMessage.id)
      ? sessionMessages
      : [...sessionMessages, userMessage];
    const placeholderId = `assistant-placeholder-${Date.now()}`;
    const optimisticMessages: Message[] = [
      ...optimisticUserMessages,
      {
        id: placeholderId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
      },
    ];

    queryClient.setQueryData(chatKeys.session(resolvedSessionId), optimisticMessages);

    const history = optimisticUserMessages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    const result = await sendMessageMutation.mutateAsync({
      message: text,
      history,
      sessionId: resolvedSessionId,
    });

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: result.text,
      timestamp: new Date(),
      stats: result.stats,
      backendConnected: result.backendConnected,
    };

    queryClient.setQueryData(
      chatKeys.session(resolvedSessionId),
      (previous: Message[] | undefined) => {
        const base = previous ?? optimisticMessages;
        return base.map((message) => (message.id === placeholderId ? assistantMessage : message));
      },
    );

    if (result.backendConnected) {
      void persistTurnMutation
        .mutateAsync({
          sessionId: resolvedSessionId,
          prompt: text,
          responseText: assistantMessage.content,
        })
        .then(() => queryClient.invalidateQueries({ queryKey: chatKeys.sessions() }))
        .catch(() => {
          // Ignore persistence failures; the response is already shown to the user.
        });
    }

    if (!sessionId) {
      setDraftMessages([createInitialAssistantMessage()]);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  return (
    <section className="surge-chat-shell">
      <style>{chatStyles}</style>

      <header className="surge-chat-header">
        <div className="surge-chat-header-icon">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
        </div>
        <div className="surge-chat-header-text">
          <h1>Surge Assistant</h1>
          <p>Disaster assessment chat</p>
        </div>
        <button
          className="surge-chat-sessions-toggle"
          type="button"
          onClick={() => setIsSessionsOpen((open) => !open)}
          aria-expanded={isSessionsOpen}
          aria-controls="surge-chat-sessions-drawer"
        >
          Sessions
        </button>
        <div className="surge-chat-status-dot" title="Online" />
      </header>

      <aside
        id="surge-chat-sessions-drawer"
        className={`surge-chat-sessions-drawer ${isSessionsOpen ? "is-open" : ""}`}
        aria-hidden={!isSessionsOpen}
      >
        <div className="surge-chat-sessions-head">
          <h2>Sessions</h2>
          <button
            className="surge-chat-new-session-btn"
            type="button"
            onClick={() => void handleCreateSession()}
          >
            New
          </button>
        </div>

        <div className="surge-chat-sessions-list">
          {sessions.length === 0 ? (
            <p className="surge-chat-empty-session-copy">No sessions yet.</p>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                className={`surge-chat-session-item ${session.id === sessionId ? "is-active" : ""}`}
                type="button"
                onClick={() => handleSelectSession(session.id)}
              >
                <span className="surge-chat-session-id">{session.id}</span>
                <span className="surge-chat-session-time">
                  {session.createdAt.toLocaleTimeString()}
                </span>
              </button>
            ))
          )}
        </div>
      </aside>

      <div className="surge-chat-messages">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {showSuggestions ? <SuggestionChips onSelect={(text) => void sendMessage(text)} /> : null}

        {isTyping ? (
          <div className="surge-chat-typing-row">
            <div className="surge-chat-avatar surge-chat-avatar-assistant">
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
              </svg>
            </div>
            <div className="surge-chat-typing-bubble">
              <TypingIndicator />
            </div>
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>

      <div className="surge-chat-input-area">
        <div className="surge-chat-input-row">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about damage levels, structure counts, and more..."
            rows={1}
          />
          <button
            className="surge-chat-send-btn"
            onClick={() => void sendMessage()}
            disabled={!input.trim() || isTyping}
            aria-label="Send message"
            type="button"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
            >
              <path d="M22 2L11 13" />
              <path d="M22 2L15 22 11 13 2 9l20-7z" />
            </svg>
          </button>
        </div>
      </div>
    </section>
  );
}

const chatStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Recursive:wght@300;400;500;600;700&display=swap');

  .surge-chat-shell,
  .surge-chat-shell *,
  .surge-chat-shell *::before,
  .surge-chat-shell *::after {
    box-sizing: border-box;
  }

  .surge-chat-shell {
    --surge-chat-bg: #0d0f14;
    --surge-chat-surface: #13161d;
    --surge-chat-border: #1f2430;
    --surge-chat-accent: #2563eb;
    --surge-chat-accent-2: #60a5fa;
    --surge-chat-text: #e8eaf0;
    --surge-chat-muted: #5a6070;
    --surge-chat-bot-bg: #1a1e28;
    --surge-chat-radius: 14px;
    --surge-chat-font: 'Recursive', sans-serif;
    --surge-chat-mono: 'Recursive', monospace;

    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    min-height: 0;
    overflow: hidden;
    background:
      radial-gradient(circle at top, rgba(37, 99, 235, 0.18), transparent 28%),
      linear-gradient(180deg, rgba(19, 22, 29, 0.98), rgba(10, 12, 18, 0.98));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    color: var(--surge-chat-text);
    font-family: var(--surge-chat-font);
  }

  .surge-chat-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 18px;
    border-bottom: 1px solid var(--surge-chat-border);
    background: rgba(15, 18, 25, 0.72);
    backdrop-filter: blur(12px);
    flex-shrink: 0;
  }

  .surge-chat-header-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--surge-chat-accent), var(--surge-chat-accent-2));
    display: grid;
    place-items: center;
    color: #fff;
    flex-shrink: 0;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
  }

  .surge-chat-header-text {
    min-width: 0;
  }

  .surge-chat-header-text h1 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  .surge-chat-header-text p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--surge-chat-muted);
  }

  .surge-chat-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #4ade80;
    margin-left: auto;
    box-shadow: 0 0 6px rgba(74, 222, 128, 0.55);
    animation: surge-chat-pulse 2s ease-in-out infinite;
    flex-shrink: 0;
  }

  .surge-chat-sessions-toggle {
    margin-left: auto;
    border: 1px solid rgba(148, 163, 184, 0.45);
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.72);
    color: var(--surge-chat-text);
    font-size: 12px;
    line-height: 1;
    padding: 8px 12px;
    cursor: pointer;
  }

  .surge-chat-sessions-drawer {
    position: absolute;
    top: 58px;
    right: 10px;
    width: min(280px, calc(100% - 20px));
    max-height: 54%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 12px;
    background: rgba(9, 12, 18, 0.95);
    transform: translateX(calc(100% + 16px));
    transition: transform 0.25s ease;
    z-index: 20;
  }

  .surge-chat-sessions-drawer.is-open {
    transform: translateX(0);
  }

  .surge-chat-sessions-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.22);
  }

  .surge-chat-sessions-head h2 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  .surge-chat-new-session-btn {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 8px;
    background: rgba(37, 99, 235, 0.18);
    color: var(--surge-chat-text);
    font-size: 11px;
    padding: 5px 8px;
    cursor: pointer;
  }

  .surge-chat-new-session-btn:disabled {
    opacity: 0.6;
    cursor: wait;
  }

  .surge-chat-sessions-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    overflow-y: auto;
    padding: 10px;
  }

  .surge-chat-empty-session-copy {
    margin: 0;
    color: var(--surge-chat-muted);
    font-size: 12px;
  }

  .surge-chat-session-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    width: 100%;
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.72);
    color: var(--surge-chat-text);
    padding: 8px;
    cursor: pointer;
    text-align: left;
  }

  .surge-chat-session-item.is-active {
    border-color: rgba(96, 165, 250, 0.72);
    background: rgba(37, 99, 235, 0.18);
  }

  .surge-chat-session-id {
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
    font-family: var(--surge-chat-mono);
  }

  .surge-chat-session-time {
    font-size: 10px;
    color: var(--surge-chat-muted);
  }

  .surge-chat-messages {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 18px 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scrollbar-width: thin;
    scrollbar-color: var(--surge-chat-border) transparent;
  }

  .surge-chat-message-row,
  .surge-chat-typing-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    animation: surge-chat-fade-up 0.25s ease both;
  }

  .surge-chat-message-row.is-user {
    flex-direction: row-reverse;
  }

  .surge-chat-avatar {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    flex-shrink: 0;
  }

  .surge-chat-avatar-assistant {
    background: var(--surge-chat-border);
    color: var(--surge-chat-accent-2);
  }

  .surge-chat-avatar-user {
    background: var(--surge-chat-accent);
    color: #fff;
  }

  .surge-chat-bubble-wrapper {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-width: 72%;
  }

  .surge-chat-message-row.is-user .surge-chat-bubble-wrapper {
    align-items: flex-end;
  }

  .surge-chat-bubble {
    padding: 11px 15px;
    border-radius: var(--surge-chat-radius);
    font-size: 14.5px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .surge-chat-bubble.is-user {
    background: var(--surge-chat-accent);
    color: #fff;
    border-bottom-right-radius: 4px;
  }

  .surge-chat-bubble.is-assistant {
    background: var(--surge-chat-bot-bg);
    color: var(--surge-chat-text);
    border: 1px solid var(--surge-chat-border);
    border-bottom-left-radius: 4px;
  }

  .surge-chat-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 4px;
  }

  .surge-chat-meta.is-user {
    justify-content: flex-end;
  }

  .surge-chat-timestamp {
    font-size: 10.5px;
    color: var(--surge-chat-muted);
    font-family: var(--surge-chat-mono);
  }

  .surge-chat-stat-cards {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
  }

  .surge-chat-stat-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid;
    border-radius: 6px;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .surge-chat-stat-num {
    font-size: 18px;
    font-weight: 700;
    font-family: var(--surge-chat-mono);
    font-variation-settings: 'MONO' 1;
  }

  .surge-chat-stat-label {
    font-size: 10px;
    color: var(--surge-chat-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: var(--surge-chat-mono);
    font-variation-settings: 'MONO' 1;
  }

  .surge-chat-suggestions {
    padding: 0 4px 2px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .surge-chat-chip {
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--surge-chat-border);
    border-radius: 999px;
    color: var(--surge-chat-muted);
    font-size: 12px;
    font-family: var(--surge-chat-font);
    cursor: pointer;
    transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  }

  .surge-chat-chip:hover {
    background: rgba(37, 99, 235, 0.16);
    border-color: rgba(37, 99, 235, 0.42);
    color: var(--surge-chat-accent-2);
    transform: translateY(-1px);
  }

  .surge-chat-typing-bubble {
    background: var(--surge-chat-bot-bg);
    border: 1px solid var(--surge-chat-border);
    border-radius: var(--surge-chat-radius);
    border-bottom-left-radius: 4px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
  }

  .surge-chat-typing-indicator {
    display: flex;
    gap: 5px;
    align-items: center;
  }

  .surge-chat-typing-indicator span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--surge-chat-muted);
    animation: surge-chat-bounce 1.2s ease-in-out infinite;
  }

  .surge-chat-typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
  }

  .surge-chat-typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
  }

  .surge-chat-input-area {
    padding: 14px 16px 16px;
    border-top: 1px solid var(--surge-chat-border);
    background: rgba(12, 15, 21, 0.88);
    flex-shrink: 0;
  }

  .surge-chat-input-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    background: var(--surge-chat-bg);
    border: 1px solid var(--surge-chat-border);
    border-radius: var(--surge-chat-radius);
    padding: 10px 10px 10px 14px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  .surge-chat-input-row:focus-within {
    border-color: var(--surge-chat-accent);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
  }

  .surge-chat-input-row textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    resize: none;
    font-family: var(--surge-chat-font);
    font-size: 14.5px;
    color: var(--surge-chat-text);
    line-height: 1.5;
    min-height: 24px;
  }

  .surge-chat-input-row textarea::placeholder {
    color: var(--surge-chat-muted);
  }

  .surge-chat-send-btn {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: var(--surge-chat-accent);
    border: none;
    cursor: pointer;
    display: grid;
    place-items: center;
    color: #fff;
    flex-shrink: 0;
    transition: background 0.2s ease, transform 0.1s ease;
  }

  .surge-chat-send-btn:hover {
    background: var(--surge-chat-accent-2);
  }

  .surge-chat-send-btn:active {
    transform: scale(0.93);
  }

  .surge-chat-send-btn:disabled {
    background: var(--surge-chat-border);
    cursor: not-allowed;
  }

  @keyframes surge-chat-pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.4;
    }
  }

  @keyframes surge-chat-fade-up {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes surge-chat-bounce {
    0%,
    80%,
    100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-6px);
      background: var(--surge-chat-accent-2);
    }
  }
`;
