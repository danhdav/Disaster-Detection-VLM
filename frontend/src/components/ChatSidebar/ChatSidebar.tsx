import classes from "./ChatSidebar.module.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  chatKeys,
  createInitialAssistantMessage,
  normalizeHistoryToMessages,
  type AllSessionHistoryMap,
  type ChatMessage as Message,
} from "../../lib/chatApi";
import {
  useChatSessionQuery,
  useChatSessionsQuery,
  useCreateChatSessionMutation,
  usePersistSessionTurnMutation,
  useSendChatMessageMutation,
} from "../../hooks/useChatQueries";

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
    <div className={classes.surgeChatSuggestions}>
      {SUGGESTIONS.map((suggestion) => (
        <button
          key={suggestion}
          className={classes.surgeChatChip}
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
    <div className={classes.surgeChatTypingIndicator}>
      <span />
      <span />
      <span />
    </div>
  );
}

function StatCards({ stats }: { stats: Record<string, string | number> }) {
  return (
    <div className={classes.surgeChatStatCards}>
      {Object.entries(stats).map(([key, value]) => (
        <div
          key={key}
          className={classes.surgeChatStatCard}
          style={{ borderLeftColor: DAMAGE_COLORS[key] ?? "#6b7280" }}
        >
          <span
            className={classes.surgeChatStatNum}
            style={{ color: DAMAGE_COLORS[key] ?? "#e5e7eb" }}
          >
            {value}
          </span>
          <span className={classes.surgeChatStatLabel}>{key.replace(/_/g, " ")}</span>
        </div>
      ))}
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div
      className={`${classes.surgeChatMessageRow} ${isUser ? classes.isUser : classes.isAssistant}`}
    >
      {!isUser ? (
        <div className={`${classes.surgeChatAvatar} ${classes.surgeChatAvatarAssistant}`}>
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

      <div className={classes.surgeChatBubbleWrapper}>
        <div
          className={`${classes.surgeChatBubble} ${isUser ? classes.isUser : classes.isAssistant}`}
        >
          {message.content}
          {message.stats ? <StatCards stats={message.stats} /> : null}
        </div>
        <div className={`${classes.surgeChatMeta} ${isUser ? classes.isUser : ""}`}>
          <span className={classes.surgeChatTimestamp}>
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>
      </div>

      {isUser ? (
        <div className={`${classes.surgeChatAvatar} ${classes.surgeChatAvatarUser}`}>
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
    <section className={classes.surgeChatShell}>
      <header className={classes.surgeChatHeader}>
        <div className={classes.surgeChatHeaderIcon}>
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
        <div className={classes.surgeChatHeaderText}>
          <h1>Surge Assistant</h1>
          <p>Disaster assessment chat</p>
        </div>
        <button
          className={classes.surgeChatSessionsToggle}
          type="button"
          onClick={() => setIsSessionsOpen((open) => !open)}
          aria-expanded={isSessionsOpen}
          aria-controls="surge-chat-sessions-drawer"
        >
          Sessions
        </button>
        <div className={classes.surgeChatStatusDot} title="Online" />
      </header>

      <aside
        id="surge-chat-sessions-drawer"
        className={`${classes.surgeChatSessionsDrawer} ${isSessionsOpen ? classes.isOpen : ""}`}
        aria-hidden={!isSessionsOpen}
      >
        <div className={classes.surgeChatSessionsHead}>
          <h2>Sessions</h2>
          <button
            className={classes.surgeChatNewSessionBtn}
            type="button"
            onClick={() => void handleCreateSession()}
          >
            New
          </button>
        </div>

        <div className={classes.surgeChatSessionsList}>
          {sessions.length === 0 ? (
            <p className={classes.surgeChatEmptySessionCopy}>No sessions yet.</p>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                className={`${classes.surgeChatSessionItem} ${session.id === sessionId ? classes.isActive : ""}`}
                type="button"
                onClick={() => handleSelectSession(session.id)}
              >
                <span className={classes.surgeChatSessionId}>{session.id}</span>
                <span className={classes.surgeChatSessionTime}>
                  {session.createdAt.toLocaleTimeString()}
                </span>
              </button>
            ))
          )}
        </div>
      </aside>

      <div className={classes.surgeChatMessages}>
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {showSuggestions ? <SuggestionChips onSelect={(text) => void sendMessage(text)} /> : null}

        {isTyping ? (
          <div className={classes.surgeChatTypingRow}>
            <div className={`${classes.surgeChatAvatar} ${classes.surgeChatAvatarAssistant}`}>
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
            <div className={classes.surgeChatTypingBubble}>
              <TypingIndicator />
            </div>
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>

      <div className={classes.surgeChatInputArea}>
        <div className={classes.surgeChatInputRow}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about damage levels, structure counts, and more..."
            rows={1}
          />
          <button
            className={classes.surgeChatSendBtn}
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
