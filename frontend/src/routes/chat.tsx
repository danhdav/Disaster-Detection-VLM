import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef, useEffect } from "react";
import { API_BASE } from "../lib/api";

export const Route = createFileRoute("/chat")({
  component: ChatPage,
});

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  stats?: Record<string, string | number> | null;
}

const SUGGESTIONS = [
  "How many structures were destroyed?",
  "How many buildings show major damage?",
  "How many buildings were undamaged?",
  "What steps are recommended next?",
];

// ----------- API Logic ------------------------------------------------------

const MOCK_RESPONSES: {
  match: RegExp;
  text: string;
  stats: Record<string, string | number> | null;
}[] = [
  {
    match: /destroy|destroyed/i,
    text: "In the current scene, **57 structures** were classified as destroyed, making 23% of all assessed buildings.",
    stats: { destroyed: 57, total_assessed: 204 },
  },
  {
    match: /major.?damage|damage.*major/i,
    text: "**31 buildings** show signs of major damage in the selected area.",
    stats: { major_damage: 31 },
  },
  {
    match: /minor.?damage|damage.*minor/i,
    text: "**18 structures** show signs of minor damage, mostly in the northeastern quadrant.",
    stats: { minor_damage: 18 },
  },
  {
    match: /no.?damage|undamaged|percent/i,
    text: "Around 54% of the assessed buildings, **108 structures** show no damage.",
    stats: { no_damage: 108 },
  },
  {
    match: /fema|next step|recommend/i,
    text: "Based on FEMA protocals, FEMA recommends prioritizing **destroyed** or **major-damage** structures for immediate inspection.",
    stats: null,
  },
];

const DAMAGE_COLORS: Record<string, string> = {
  destroyed: "#ef4444",
  major_damage: "#f97316",
  minor_damage: "#eab308",
  no_damage: "#22c55e",
  un_classified: "#6b7280",
  total_assessed: "#3b82f6",
};

async function sendChatMessage(
  message: string,
  history: { role: Role; content: string }[],
): Promise<{
  text: string;
  stats: Record<string, string | number> | null;
}> {
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        conversation_history: history,
      }),
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return {
      text: data.response ?? data.message,
      stats: data.stats ?? null,
    };
  } catch {
    await new Promise((r) => setTimeout(r, 800));

    const match = MOCK_RESPONSES.find((r) => r.match.test(message.toLowerCase()));
    return {
      text:
        match?.text ??
        "I can help you query disaster assessment data. Try asking about destroyed structures, damage levels, or scene statistics.",
      stats: match?.stats ?? null,
    };
  }
}

// ----------- Components ------------------------------------------------------

function SuggestionChips({ onSelect }: { onSelect: (text: string) => void }) {
  return (
    <div className="suggestions">
      {SUGGESTIONS.map((s) => (
        <button key={s} className="chip" onClick={() => onSelect(s)}>
          {s}
        </button>
      ))}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span />
      <span />
      <span />
    </div>
  );
}

function RenderContent({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**"))
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith("`") && part.endsWith("`"))
          return (
            <code key={i} className="inline-code">
              {part.slice(1, -1)}
            </code>
          );
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function StatCards({ stats }: { stats: Record<string, string | number> }) {
  return (
    <div className="stat-cards">
      {Object.entries(stats).map(([key, val]) => (
        <div
          key={key}
          className="stat-card"
          style={{
            borderLeftColor: DAMAGE_COLORS[key] ?? "#6b7280",
          }}
        >
          <span className="stat-num" style={{ color: DAMAGE_COLORS[key] ?? "#e5e7eb" }}>
            {val}
          </span>
          <span className="stat-label">{key.replace(/_/g, " ")}</span>
        </div>
      ))}
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="bubble-wrapper">
        <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
          {isUser ? message.content : <RenderContent text={message.content} />}
          {message.stats && <StatCards stats={message.stats} />}
        </div>
        <div className={`msg-meta ${isUser ? "user" : ""}`}>
          <span className="timestamp">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>
      </div>
    </div>
  );
}

//----------- Page Component ---------------------------------------------------------

function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "0",
      role: "assistant",
      content:
        "Hi, I am your disaster assessment assistant. Select a question below or type your own question.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const sendMessage = async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || isTyping) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);
    setShowSuggestions(false);

    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    const result = await sendChatMessage(text, history);

    setIsTyping(false);
    setMessages((prev) => [
      ...prev,
      {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: result.text,
        timestamp: new Date(),
        stats: result.stats,
      },
    ]);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
  }, [input]);

  return (
    <>
      <style>{chatStyles}</style>

      <div className="chat-shell">
        {/* Header */}
        <header className="chat-header">
          <div className="header-icon">
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
          <div className="header-text">
            <h1>Surge Assistant</h1>
          </div>
          <div className="status-dot" title="Online" />
          <button
            className="send-btn clear-btn"
            onClick={() => {
              setMessages([
                {
                  id: "0",
                  role: "assistant",
                  content:
                    "Hi, I am your disaster assessment assistant. Select a question below or type your own question.",
                  timestamp: new Date(),
                },
              ]);
              setShowSuggestions(true);
            }}
            title="Clear chat"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="lucide lucide-rotate-ccw-icon lucide-rotate-ccw"
            >
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
          </button>
        </header>

        {/* Messages */}
        <div className="messages-area">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {showSuggestions && (
            <SuggestionChips
              onSelect={(text) => {
                void sendMessage(text);
              }}
            />
          )}

          {isTyping && (
            <div className="typing-row">
              <div className="typing-bubble">
                <TypingIndicator />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="input-area">
          <div className="input-row">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about damage levels, structure counts, and more..."
              rows={1}
            />
            <button
              className="send-btn"
              onClick={() => void sendMessage()}
              disabled={!input.trim() || isTyping}
              aria-label="Send message"
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
      </div>
    </>
  );
}

// ----------- Styles --------------------------------------------------
const chatStyles = `
    @import url('https://fonts.googleapis.com/css2?family=Recursive:wght@300;400;500;600;700&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --bg:        #0d1117;
        --surface:   rgba(255, 255, 255, 0.05);
        --border:    rgba(255, 255, 255, 0.1);
        --accent:    #3b7ef8;
        --accent2:   #7eb3ff;
        --text:      #e8edf5;
        --muted:     #7a8aaa;
        --bot-bg:    rgba(255, 255, 255, 0.06);
        --radius:    14px;
        --font:      'Recursive', sans-serif;
        --mono:      'Recursive', monospace;
    }

    body { background: var(--bg); font-family: var(--font); color: var(--text); }

    /* ── Suggestion chips ── */
    .suggestions {
        padding: 0 20px 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .chip {
        padding: 5px 12px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.01);
        border-radius: 20px;
        color: var(--muted);
        font-size: 12px;
        font-family: var(--font);
        cursor: pointer;
        transition: all 0.15s;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .chip:hover {
        background: rgba(37, 99, 235, 0.15);
        border-color: rgba(37, 99, 235, 0.4);
        color: var(--accent2);
    }

    /* ── Inline code ── */
    .inline-code {
        background: rgba(59, 126, 248, 0.15);
        color: var(--accent2);
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 0.85em;
        font-family: var(--mono);
    }

    /* ── Message meta row ── */
    .msg-meta {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 0 4px;
    }
    .msg-meta.user { justify-content: flex-end; }

    /* ── Stat cards ── */
    .stat-cards {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.01);
        border-left: 3px solid;
        border-radius: 6px;
        padding: 6px 10px;
        display: flex;
        flex-direction: column;
        gap: 1px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .stat-num {
        font-size: 18px;
        font-weight: 700;
        font-family: var(--mono);
        font-variation-settings: "MONO" 1;
    }
    .stat-label {
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: var(--mono);
        font-variation-settings: "MONO" 1;
    }

    .chat-shell {
        display: flex;
        flex-direction: column;
        height: 100dvh;
        max-width: 820px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── Header ── */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 18px 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .header-icon {
        width: 38px; height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        display: grid; place-items: center;
        color: #fff;
        flex-shrink: 0;
    }
    .header-text h1 {
        font-size: 15px; font-weight: 600; letter-spacing: -0.01em;
    }
    .header-text p {
        font-size: 12px; color: var(--muted); font-family: var(--mono);
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #4ade80;
        margin-left: auto;
        box-shadow: 0 0 6px #4ade8088;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
    }

    /* ── Messages ── */
    .messages-area {
        flex: 1;
        overflow-y: auto;
        padding: 24px 20px;
        display: flex;
        flex-direction: column;
        gap: 18px;
        scrollbar-width: thin;
        scrollbar-color: var(--border) transparent;
    }

    .message-row {
        display: flex;
        gap: 10px;
        align-items: flex-end;
        animation: fadeUp 0.25s ease both;
    }
    .message-row.user { flex-direction: row-reverse; }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .bubble-wrapper { display: flex; flex-direction: column; gap: 4px; max-width: 72%; }
    .message-row.user .bubble-wrapper { align-items: flex-end; }

    .bubble {
        padding: 11px 15px;
        border-radius: var(--radius);
        font-size: 14.5px;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .bubble-user {
        background: var(--accent);
        color: #fff;
        border-bottom-right-radius: 4px;
    }
    .bubble-assistant {
        background: rgba(255, 255, 255, 0.06);
        color: var(--text);
        border: 1px solid var(--border);
        border-bottom-left-radius: 4px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    .timestamp {
        font-size: 10.5px;
        color: var(--muted);
        font-family: var(--mono);
        padding: 0 4px;
    }

    /* ── Typing indicator ── */
    .typing-row {
        display: flex; gap: 10px; align-items: flex-end;
        animation: fadeUp 0.2s ease both;
    }
    .typing-bubble {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius);
        border-bottom-left-radius: 4px;
        padding: 14px 16px;
        display: flex; align-items: center;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .typing-indicator {
        display: flex; gap: 5px; align-items: center;
    }
    .typing-indicator span {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--muted);
        animation: bounce 1.2s ease-in-out infinite;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-6px); background: var(--accent2); }
    }

    /* ── Input area ── */
    .input-area {
        padding: 16px 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .input-row {
        display: flex;
        align-items: flex-end;
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius);
        padding: 10px 10px 10px 16px;
        transition: border-color 0.2s;
    }
    .input-row:focus-within {
        border-color: rgba(59, 126, 248, 0.6);
        box-shadow: 0 0 0 3px rgba(59, 126, 248, 0.1);
    }
    textarea {
        flex: 1;
        background: transparent;
        border: none;
        outline: none;
        resize: none;
        font-family: var(--font);
        font-size: 14.5px;
        color: var(--text);
        line-height: 1.5;
        min-height: 24px;
    }
    textarea::placeholder { color: var(--muted); }

    .send-btn {
        width: 30px; height: 30px;
        border-radius: 8px;
        background: var(--accent);
        border: none;
        cursor: pointer;
        display: grid; place-items: center;
        color: #fff;
        flex-shrink: 0;
        transition: background 0.2s, transform 0.1s;
    }
    .send-btn:hover { background: var(--accent2); }
    .send-btn:active { transform: scale(0.93); }
    .send-btn:disabled { background: var(--border); cursor: not-allowed; }
    
    .clear-btn {
        background: var(--border);
        cursor: pointer;
    }
    .clear-btn:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    `;
