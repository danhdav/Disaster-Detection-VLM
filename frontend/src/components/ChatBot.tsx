"use client";

import { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send, Minimize2 } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello! I can help you understand disaster detection and damage assessment. How can I assist you today?",
    },
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    // Simulate a response (replace with actual API call later)
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Thank you for your question. This chat feature is currently in development. Soon, I'll be able to provide detailed information about disaster detection analysis and help you interpret your results.",
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }, 1000);
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 bg-[#1a3a4a] hover:bg-[#244a5a] border border-[#2a5a6a]/50 rounded-full flex items-center justify-center shadow-lg shadow-black/30 transition-all duration-300 hover:scale-105 z-50"
        aria-label="Open chat"
      >
        <MessageCircle className="w-6 h-6 text-white" />
      </button>
    );
  }

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 transition-all duration-300 ${
        isMinimized ? "w-72" : "w-80 sm:w-96"
      }`}
    >
      <div
        className={`bg-[#0d1114] border border-[#2a3038] rounded-2xl shadow-2xl shadow-black/40 overflow-hidden flex flex-col transition-all duration-300 ${
          isMinimized ? "h-14" : "h-[500px]"
        }`}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-3 bg-[#1a1f24] border-b border-[#2a3038] cursor-pointer"
          onClick={() => isMinimized && setIsMinimized(false)}
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-[#1a3a4a] border border-[#2a5a6a]/50 flex items-center justify-center">
              <MessageCircle className="w-4 h-4 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Disaster Assistant</h3>
              {!isMinimized && (
                <span className="text-xs text-muted-foreground">Online</span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsMinimized(!isMinimized);
              }}
              className="p-1.5 hover:bg-[#2a3038] rounded-lg transition-colors"
              aria-label={isMinimized ? "Expand chat" : "Minimize chat"}
            >
              <Minimize2 className="w-4 h-4 text-muted-foreground" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
              }}
              className="p-1.5 hover:bg-[#2a3038] rounded-lg transition-colors"
              aria-label="Close chat"
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </button>
          </div>
        </div>

        {/* Messages */}
        {!isMinimized && (
          <>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                      message.role === "user"
                        ? "bg-[#1a3a4a] text-white rounded-br-md"
                        : "bg-[#1a1f24] text-gray-200 rounded-bl-md border border-[#2a3038]"
                    }`}
                  >
                    {message.content}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-4 border-t border-[#2a3038]">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type a message..."
                  className="flex-1 bg-[#1a1f24] border border-[#2a3038] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-[#2a5a6a]/50 focus:border-[#2a5a6a] transition-all"
                />
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="w-10 h-10 bg-[#1a3a4a] hover:bg-[#244a5a] disabled:opacity-50 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition-colors"
                  aria-label="Send message"
                >
                  <Send className="w-4 h-4 text-white" />
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
