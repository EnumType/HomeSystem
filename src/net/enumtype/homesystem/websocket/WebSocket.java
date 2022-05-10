package net.enumtype.homesystem.websocket;

import java.io.File;
import java.io.IOException;

import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.server.WebSocketHandler;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;

import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;

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
					server.setHandler(new WebSocketHandler() {
						@Override
						public void configure(WebSocketServletFactory factory) {
							factory.register(Handler.class);
						}
					});

					HttpConfiguration http = new HttpConfiguration();
					http.setSecureScheme("https");
					http.setSecurePort(httpsPort);

					HttpConfiguration https = new HttpConfiguration(http);
					https.addCustomizer(new SecureRequestCustomizer());

					SslContextFactory ssl = new SslContextFactory.Server();
					ssl.setKeyStorePath(keystore);
					ssl.setKeyStorePassword(keystorePassword);

					ServerConnector wsConnector = new ServerConnector(server);
					wsConnector.setPort(httpPort);
					server.addConnector(wsConnector);

					ServerConnector wssConnector = new ServerConnector(server,
							new SslConnectionFactory(ssl, HttpVersion.HTTP_1_1.asString()),
							new HttpConnectionFactory(https));
					wssConnector.setPort(httpsPort);
					server.addConnector(wssConnector);

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
