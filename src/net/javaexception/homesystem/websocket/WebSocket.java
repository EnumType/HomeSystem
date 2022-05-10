package net.javaexception.homesystem.websocket;

import java.io.File;
import java.io.IOException;

import org.eclipse.jetty.server.Connector;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.eclipse.jetty.webapp.WebAppContext;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.server.WebSocketHandler;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;

import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

public class WebSocket {
	
	public static void startWebSocket(int httpPort, int httpsPort, String keystoredir, String keystore, String keystorePassword) throws IOException {
		File file = new File(keystoredir);
		File keystoreFile = new File(keystore);
		
		if(file.exists() && keystoreFile.exists()) {
			Thread thread = new Thread(() -> {
				try {
					System.setProperty("org.eclipse.jetty.LEVEL", "OFF");
					System.setProperty("org.eclipse.jetty.util.log.class", "org.eclipse.jetty.util.log.StdErrLog");
					Log.write(Methods.createPrefix() + "Starting Websocket: Http:" + httpPort + " Https:" + httpsPort, true);

					Server server = new Server();

					WebAppContext webapp = new WebAppContext();
					webapp.setResourceBase("src/main/webapp");
					server.setHandler(webapp);

					HttpConfiguration http = new HttpConfiguration();
					http.addCustomizer(new SecureRequestCustomizer());
					http.setSecurePort(httpsPort);
					http.setSecureScheme("https");

					ServerConnector connector = new ServerConnector(server);
					connector.addConnectionFactory(new HttpConnectionFactory(http));
					connector.setPort(httpPort);

					HttpConfiguration https = new HttpConfiguration();
					https.addCustomizer(new SecureRequestCustomizer());

					SslContextFactory ssl = new SslContextFactory.Server();
					ssl.setKeyStorePath(keystore);
					ssl.setKeyStorePassword(keystorePassword);

					ServerConnector sslConnector = new ServerConnector(server, new SslConnectionFactory(ssl, "http/1.1"), new HttpConnectionFactory(https));
					sslConnector.setPort(httpsPort);

					WebSocketHandler wsHandler = new WebSocketHandler() {
						@Override
						public void configure(WebSocketServletFactory factory) {
							factory.register(Handler.class);
						}
					};

					server.setConnectors(new Connector[]{connector, sslConnector});
					server.setHandler(wsHandler);

					server.start();
					server.join();
				}catch(Exception e) {
					e.printStackTrace();
					Log.write(Methods.createPrefix() + "Error in WebSocket(65): " + e.getMessage(), false);
				}
			});
			
			thread.start();
		}else {
			if(!file.exists()) if(!file.mkdir()) throw new IOException("Cannot create directory!");
			Log.write(Methods.createPrefix() + "No Keystore Found! Don't start Weboscket!", false);
		}
	}
	
	public static void sendCommand(String command, Session session) {
		try {
			session.getRemote().sendString(command);
		}catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void closeConnection(Session session) {
		session.close();
	}
	
}
