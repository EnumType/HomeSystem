package net.enumtype.homesystem.test;

import org.eclipse.jetty.server.*;
import org.eclipse.jetty.websocket.server.WebSocketHandler;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;

public class TestMain {

    public static void main(String[] args) {
        try {
            //startTestWebSocket();
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static void startTestWebSocket() {
        try {
            System.setProperty("org.eclipse.jetty.LEVEL", "OFF");
            System.setProperty("org.eclipse.jetty.util.log.class", "org.eclipse.jetty.util.log.StdErrLog");

            Server server = new Server();
            server.setHandler(new WebSocketHandler() {
                @Override
                public void configure(WebSocketServletFactory factory) {
                    factory.register(TestWebSocketHandler.class);
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

}