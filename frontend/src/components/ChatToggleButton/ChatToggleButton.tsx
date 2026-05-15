import classes from "./ChatToggleButton.module.css";

interface ChatToggleButtonProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function ChatToggleButton({ isOpen, onToggle }: ChatToggleButtonProps) {
  return (
    <button
      className={classes.chatToggleBtn}
      onClick={onToggle}
      aria-label={isOpen ? "Close chat sidebar" : "Open chat sidebar"}
      type="button"
      title={isOpen ? "Close Surge Assistant" : "Open Surge Assistant"}
    >
      {isOpen ? (
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
  );
}
