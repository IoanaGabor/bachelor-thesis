const getWebsocketUrl = () => {
  const apiUrl = new URL(import.meta.env.VITE_PUBLIC_ENDPOINT);
  const protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${apiUrl.host}/notifications/ws/`;
};

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private listeners: { [key: string]: ((data: any) => void)[] } = {};

  connect(personId: string) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = `${getWebsocketUrl()}${personId}`;
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        console.log('WebSocket connection established');
        this.reconnectAttempts = 0;
      };

      this.socket.onmessage = (event) => {
        console.log("WebSocket message received:", event.data);
        try {
          const data = JSON.parse(event.data);
          console.log("WebSocket message received:", data);
          if (data.type === 'reconstruction_notification') {
    
            this.notifyListeners('reconstruction_notification', data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.socket.onclose = (event) => {
        if (!event.wasClean) {
          console.log('WebSocket connection closed unexpectedly. Attempting to reconnect...');
          this.attemptReconnect(personId);
        } else {
          console.log('WebSocket connection closed cleanly');
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
    }
  }

  private attemptReconnect(personId: string) {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Maximum reconnect attempts reached');
      return;
    }

    const delay = Math.min(1000 * (2 ** this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    
    this.reconnectTimeout = setTimeout(() => {
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      this.connect(personId);
    }, delay);
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  addListener(event: string, callback: (data: any) => void) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  removeListener(event: string, callback: (data: any) => void) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  private notifyListeners(event: string, data: any) {
    if (this.listeners[event]) {
      console.log("Notifying listeners for event:", event, "with data:", data, "listeners:", this.listeners[event]);
      this.listeners[event].forEach(callback => callback(data));
    }
  }
}

export const websocketService = new WebSocketService(); 