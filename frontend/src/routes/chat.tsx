import { createFileRoute } from "@tanstack/react-router";

import { ChatSidebar } from "../components/ChatSidebar/ChatSidebar";

export const Route = createFileRoute("/chat")({
  component: ChatPage,
});

function ChatPage() {
  return (
    <div
      style={{
        height: "100dvh",
        padding: "16px",
        boxSizing: "border-box",
        background: "#020917",
        overflow: "hidden",
      }}
    >
      <ChatSidebar />
    </div>
  );
}