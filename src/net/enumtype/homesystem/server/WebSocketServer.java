package net.enumtype.homesystem.server;

import java.io.File;
import java.io.IOException;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.utils.Data;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.UnknownCommandException;
import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.SecureRequestCustomizer;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.util.log.StdErrLog;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.*;
import org.eclipse.jetty.websocket.server.WebSocketHandler;
import org.eclipse.jetty.websocket.servlet.WebSocketServletFactory;

@WebSocket(maxIdleTime=120000)
public class WebSocketServer {

	private final int httpPort;
	private final int httpsPort;
	private final String keystoreDir;
	private final String keystore;
	private final String keystorePassword;
	private final Log log;
	private Server server;

	public WebSocketServer() {
		final Data data = Main.getData();
		this.httpPort = data.getWsPort();
		this.httpsPort = data.getWssPort();
		this.keystoreDir = data.getResourcesDir();
		this.keystore = data.getResourcesDir() + "//" + data.getWsKeystore();
		this.keystorePassword = data.getWsKeystorePassword();
		this.log = Main.getLog();
	}

	public void start() throws IOException {
		File file = new File(keystoreDir);
		File keystoreFile = new File(keystore);
		
		if(file.exists() && keystoreFile.exists()) {
			Thread thread = new Thread(() -> {
				try {
					final StdErrLog logger = new StdErrLog();
					logger.setLevel(StdErrLog.LEVEL_OFF);
					org.eclipse.jetty.util.log.Log.setLog(logger);

					log.write("Starting WebSocket: Http:" + httpPort + " Https:" + httpsPort);

					server = new Server();
					server.setHandler(new WebSocketHandler() {
						@Override
						public void configure(WebSocketServletFactory factory) {
							factory.register(WebSocketServer.class);
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

					System.out.print(Main.getCommandPrefix());
					server.start();
					server.join();
				}catch(Exception e) {
					log.writeError(e);
				}
			});
			
			thread.start();
		}else {
			if(!file.exists()) if(!file.mkdir()) throw new IOException("Cannot create directory!");
			log.write("No Keystore Found! Don't start WebSocket!");
		}
	}

	public void stop() throws Exception {
		if(server != null) server.stop();
	}

	//WebSocketHandler
	@OnWebSocketClose
	public void onClose(Session session, int statusCode, String reason) {
		final ClientManager clientManager = Main.getClientManager();

		if(clientManager.isLoggedIn(session) && !clientManager.isChangingConnection(session))
			clientManager.getClient(session).logout();
	}

	@OnWebSocketError
	public void onError(Session session, Throwable t) {
		log.writeError(t);
		ClientManager clientManager = Main.getClientManager();

		if(clientManager.isLoggedIn(session)) clientManager.getClient(session).logout();
	}

	@OnWebSocketConnect
	public void onConnect(Session session) {
		try {
			final ClientManager clientManager = Main.getClientManager();
			if(!clientManager.isLoggedIn(session)) {
				session.getRemote().sendString("notloggedin");
			}else {
				final Client client = clientManager.getClient(session.getRemoteAddress().getAddress());
				client.setSession(session);
				client.sendMessage("loggedin");
				client.sendMessage("user:" + client.getName());
			}
		}catch(IOException e) {
			log.writeError(e);
		}
	}

	@OnWebSocketMessage
	public void onMessage(Session session, String message) {
		if(message.equalsIgnoreCase("isonline")) return;

		try {
			Command.check(message, session);
		}catch(UnknownCommandException e) {
			Main.getLog().writeError(e);
		}
	}
	
}