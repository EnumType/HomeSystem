package net.enumtype.homesystem.websocket;

import java.io.IOException;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.server.ClientManager;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketClose;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketConnect;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketError;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketMessage;
import org.eclipse.jetty.websocket.api.annotations.WebSocket;

import net.enumtype.homesystem.server.Command;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;

@WebSocket(maxIdleTime=120000)
public class Handler {
	
	@OnWebSocketClose
	public void onClose(Session session, int statusCode, String reason) {
		ClientManager clientManager = Main.getClientManager();

		if(clientManager.isLoggedIn(session)) clientManager.logoutClient(clientManager.getClient(session));
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
			}
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Handler(88): " + e.getMessage(), true);
			Log.write("", false);
		}
	}
	
	@OnWebSocketMessage
	public void onMessage(Session session, String message) {
		Command.check(message, session);
	}
	
}
