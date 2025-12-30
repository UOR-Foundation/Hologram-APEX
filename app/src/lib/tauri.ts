import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';

export interface Model {
  name: string;
  size: number;
  digest: string;
  modified_at: string;
  details?: {
    format: string;
    family: string;
    parameter_size: string;
    quantization_level: string;
  };
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface StreamChunk {
  content: string;
  done: boolean;
}

interface PullProgress {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
}

export async function listModels(): Promise<Model[]> {
  return invoke('list_models');
}

export async function sendMessage(
  model: string,
  messages: ChatMessage[],
  onChunk: (content: string, done: boolean) => void
): Promise<void> {
  let unlisten: UnlistenFn | null = null;

  try {
    unlisten = await listen<StreamChunk>('chat-response', (event) => {
      onChunk(event.payload.content, event.payload.done);
    });

    await invoke('send_message', { model, messages });
  } finally {
    if (unlisten) {
      unlisten();
    }
  }
}

export async function pullModel(
  name: string,
  onProgress: (progress: PullProgress) => void
): Promise<void> {
  const unlisten = await listen<PullProgress>('pull-progress', (event) => {
    onProgress(event.payload);
  });

  try {
    await invoke('pull_model', { name });
  } finally {
    unlisten();
  }
}

export async function deleteModel(name: string): Promise<void> {
  return invoke('delete_model', { name });
}

export async function getVersion(): Promise<string> {
  return invoke('get_version');
}
