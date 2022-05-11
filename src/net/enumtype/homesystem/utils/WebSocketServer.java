package net.enumtype.homesystem.utils;

import java.io.File;
import java.io.IOException;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.server.ClientManager;
import net.enumtype.homesystem.server.Command;
import net.enumtype.homesystem.server.UnknownCommandException;
import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.*;
import org.eclipse.jetty.websocket.server.WebSocketHandler;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;

public class WebSocketServer {

	private final int httpPort;
	private final int httpsPort;
	private final String keystoreDir;
	private final String keystore;
	private final String keystorePassword;
	private final Log log;
	private Server server;

	public WebSocketServer(int httpPort, int httpsPort, String keystoreDir, String keystore, String keystorePassword) {
		this.httpPort = httpPort;
		this.httpsPort = httpsPort;
		this.keystoreDir = keystoreDir;
		this.keystore = keystore;
		this.keystorePassword = keystorePassword;
		this.log = Main.getLog();
	}

	public void start() throws IOException {
		File file = new File(keystoreDir);
		File keystoreFile = new File(keystore);
		
		if(file.exists() && keystoreFile.exists()) {
			Thread thread = new Thread(() -> {
				try {
					System.setProperty("org.eclipse.jetty.LEVEL", "OFF");
					System.setProperty("org.eclipse.jetty.util.log.class", "org.eclipse.jetty.util.log.StdErrLog");
					log.write(Methods.createPrefix() + "Starting Websocket: Http:" + httpPort + " Https:" + httpsPort, true);

					server = new Server();
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
					log.write(Methods.createPrefix() + "Error in WebSocket(65): " + e.getMessage(), false);
				}
			});
			
			thread.start();
		}else {
			if(!file.exists()) if(!file.mkdir()) throw new IOException("Cannot create directory!");
			log.write(Methods.createPrefix() + "No Keystore Found! Don't start Weboscket!", false);
		}
	}

	public void stop() throws Exception {
		server.stop();
	}
	
}

@WebSocket(maxIdleTime=120000)
class Handler {

	@OnWebSocketClose
	public void onClose(Session session, int statusCode, String reason) {
		final ClientManager clientManager = Main.getClientManager();

		if(clientManager.isLoggedIn(session) && !clientManager.isChangingConnection(session))
			clientManager.logoutClient(clientManager.getClient(session));
	}

	@OnWebSocketError
	public void onError(Session session, Throwable t) {
		ClientManager clientManager = Main.getClientManager();

		if(clientManager.isLoggedIn(session)) clientManager.logoutClient(clientManager.getClient(session));
	}

	@OnWebSocketConnect
	public void onConnect(Session session) {
		try {
			if(!Main.getClientManager().isLoggedIn(session)) {
				session.getRemote().sendString("notloggedin");
			}else {
				session.getRemote().sendString("loggedin");
				Main.getClientManager().getClient(session).changeConnection(false);
			}
		}catch(IOException e) {
			e.printStackTrace();
		}
	}

	@OnWebSocketMessage
	public void onMessage(Session session, String message) {
		try {
			Command.check(message, session);
		}catch(UnknownCommandException e) {
			e.printStackTrace();
		}
	}

}