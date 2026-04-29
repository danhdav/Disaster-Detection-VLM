import { useState, useMemo } from "react";
import classes from "./ChatToggleButton.module.css";
import { ChatSidebar } from "../ChatSidebar/ChatSidebar";
import { useChatSessionsQuery } from "../../hooks/useChatQueries";


export function ChatToggleButton() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const sessionsQuery = useChatSessionsQuery();

  const sessions = useMemo(() => {
    const allHistory = sessionsQuery.data ?? {};
    return Object.keys(allHistory).map((id, index) => ({
      id,
      createdAt: new Date(Date.now() - index * 1000),
    }));
  }, [sessionsQuery.data]);

  const mostRecentSessionId = sessions.length > 0 ? sessions[0].id : null;

  return (
    <>
      <button
        className={classes.chatToggleBtn}
        onClick={() => setIsChatOpen(!isChatOpen)}
        aria-label="Toggle chatbot"
        type="button"
        title="Open Surge Assistant"
      >
        {isChatOpen ? (
          // Bot-off icon for closing
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="lucide lucide-bot-off-icon lucide-bot-off"
          >
            <path d="M13.67 8H18a2 2 0 0 1 2 2v4.33" />
            <path d="M2 14h2" />
            <path d="M20 14h2" />
            <path d="M22 22 2 2" />
            <path d="M8 8H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 1.414-.586" />
            <path d="M9 13v2" />
            <path d="M9.67 4H12v2.33" />
          </svg>
        ) : (
          // Original bot icon for opening
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="lucide lucide-bot-icon lucide-bot"
          >
            <path d="M12 8V4H8" />
            <rect width="16" height="12" x="4" y="8" rx="2" />
            <path d="M2 14h2" />
            <path d="M20 14h2" />
            <path d="M15 13v2" />
            <path d="M9 13v2" />
          </svg>
        )}
      </button>

      <div className={`${classes.chatPanel} ${isChatOpen ? classes.chatPanelOpen : classes.chatPanelClosed}`}>
        <ChatSidebar initialSessionId={mostRecentSessionId} />
      </div>
    </>
  );
}