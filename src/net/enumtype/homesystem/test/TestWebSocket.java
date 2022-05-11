package net.enumtype.homesystem.test;

import net.enumtype.homesystem.utils.Methods;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.*;
import org.eclipse.jetty.websocket.server.WebSocketHandler;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;

@WebSocket(maxIdleTime=120000)
public class TestWebSocket {

    public static void start() {
        try {
            System.setProperty("org.eclipse.jetty.LEVEL", "OFF");
            System.setProperty("org.eclipse.jetty.util.log.class", "org.eclipse.jetty.util.log.StdErrLog");

            Server server = new Server();
            server.setHandler(new WebSocketHandler() {
                @Override
                public void configure(WebSocketServletFactory factory) {
                    factory.register(TestWebSocket.class);
                }
            });

            HttpConfiguration http = new HttpConfiguration();
            http.setSecureScheme("https");
            http.setSecurePort(8001);

            HttpConfiguration https = new HttpConfiguration(http);
            https.addCustomizer(new SecureRequestCustomizer());

            ServerConnector wsConnector = new ServerConnector(server);
            wsConnector.setPort(8000);
            server.addConnector(wsConnector);

            server.start();
            server.join();
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

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