package net.enumtype.homesystem.test;

import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.*;

@WebSocket(maxIdleTime=120000)
public class TestWebSocketHandler {

    @OnWebSocketClose
    public void onClose(Session session, int statusCode, String reason) {
        System.out.println("close");
    }

    @OnWebSocketError
    public void onError(Session session, Throwable t) {
        System.out.println("error");
    }

    @OnWebSocketConnect
    public void onConnect(Session session) {
        System.out.println("connect");
    }

    @OnWebSocketMessage
    public void onMessage(Session session, String message) {
        System.out.println(message);
    }

}
