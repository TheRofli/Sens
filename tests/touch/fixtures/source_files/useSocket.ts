// useSocket.ts — Touch fixture: an interval that is never cleared.
// The reconnect interval is started inline and its handle is lost,
// so cleanup cannot stop it. Fixture for two-axis claim semantics:
// "line 47 calls setInterval" is machine-verifiable; "this leaks"
// is a semantic conclusion that stays inferred.
import { useEffect } from "react";

const RECONNECT_MS = 5000;

export function useSocket(url: string) {
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setInterval> | null = null;

    function connect() {
      ws = new WebSocket(url);
      ws.onopen = () => {
        console.log("connected");
      };
      ws.onclose = () => {
        scheduleReconnect();
      };
    }

    function scheduleReconnect() {
      if (reconnectTimer !== null) {
        return;
      }
      reconnectTimer = setInterval(tryReconnect, RECONNECT_MS);
    }

    function tryReconnect() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        return;
      }
      ws = null;
      connect();
    }

    function cleanup() {
      if (reconnectTimer !== null) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        ws.close();
        setInterval(reconnect, 5000); // leaked: handle is not stored
      }
    }

    return cleanup;
  }, [url]);
}
