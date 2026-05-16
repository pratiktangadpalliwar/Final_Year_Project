import type { WsEvent } from "./types";

export class LiveSocket {
  private ws: WebSocket | null = null;
  private listeners: ((e: WsEvent) => void)[] = [];
  private retryDelayMs = 1000;

  connect() {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`;
    this.ws = new WebSocket(url);
    this.ws.onmessage = (ev) => {
      try {
        const data: WsEvent = JSON.parse(ev.data);
        this.listeners.forEach((fn) => fn(data));
      } catch (e) {
        console.warn("bad ws frame", e);
      }
    };
    this.ws.onclose = () => {
      this.ws = null;
      setTimeout(() => this.connect(), this.retryDelayMs);
    };
    this.ws.onerror = () => this.ws?.close();
  }

  subscribe(fn: (e: WsEvent) => void): () => void {
    this.listeners.push(fn);
    return () => { this.listeners = this.listeners.filter((f) => f !== fn); };
  }
}

export const liveSocket = new LiveSocket();
