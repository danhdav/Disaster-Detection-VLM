import { useMutation, useQuery } from "@tanstack/react-query";

import {
  chatKeys,
  createChatSession,
  fetchAllSessionHistory,
  fetchSessionHistory,
  normalizeHistoryToMessages,
  persistSessionTurn,
  sendChatMessage,
  type ChatMessage,
  type ChatMessageRequest,
} from "../lib/chatApi";

export function useChatSessionsQuery() {
  return useQuery({
    queryKey: chatKeys.sessions(),
    queryFn: fetchAllSessionHistory,
    retry: false,
    refetchOnWindowFocus: false,
  });
}

export function useChatSessionQuery(sessionId: string | null) {
  return useQuery({
    queryKey: sessionId ? chatKeys.session(sessionId) : chatKeys.session("pending"),
    queryFn: async (): Promise<ChatMessage[]> => {
      const history = await fetchSessionHistory(sessionId!);
      return normalizeHistoryToMessages(history);
    },
    enabled: Boolean(sessionId),
    retry: false,
    refetchOnWindowFocus: false,
  });
}

export function useCreateChatSessionMutation() {
  return useMutation({
    mutationFn: createChatSession,
  });
}

export function useSendChatMessageMutation() {
  return useMutation({
    mutationFn: ({
      message,
      history,
      sessionId,
    }: {
      message: string;
      history: ChatMessageRequest[];
      sessionId: string;
    }) => sendChatMessage(message, history, sessionId),
  });
}

export function usePersistSessionTurnMutation() {
  return useMutation({
    mutationFn: ({
      sessionId,
      prompt,
      responseText,
    }: {
      sessionId: string;
      prompt: string;
      responseText: string;
    }) => persistSessionTurn(sessionId, prompt, responseText),
  });
}
