package net.javaexception.homesystem.websocket;

import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;

import net.javaexception.homesystem.main.Main;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketClose;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketConnect;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketError;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketMessage;
import org.eclipse.jetty.websocket.api.annotations.WebSocket;

import net.javaexception.homesystem.server.Command;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

@WebSocket(maxIdleTime=120000)
public class Handler{
	
	public static HashMap<InetAddress, Session> sessions = new HashMap<>();
	
	@OnWebSocketClose
	public void onClose(Session session, int statusCode, String reason) {
		InetAddress address = session.getRemoteAddress().getAddress();
		sessions.remove(address);

		if(Main.getClientManager().isLoggedIn(address)) {
			Main.getClientManager().logoutClient(Main.getClientManager().getClient(address));
		}
	}
	
	@OnWebSocketError
	public void onError(Session session, Throwable t) {
		InetAddress address = session.getRemoteAddress().getAddress();
		sessions.remove(address);
		
		if(Main.getClientManager().isLoggedIn(address)) {
			Main.getClientManager().logoutClient(Main.getClientManager().getClient(address));
		}
	}
	
	@OnWebSocketConnect
	public void onConnect(Session session) {
		InetAddress address = session.getRemoteAddress().getAddress();

		sessions.remove(address);
		sessions.put(address, session);

		Log.write(Methods.createPrefix() + "New Connection to WebSocket Server:", true);
		Log.write("INETADDRESS: " + address, true);
		Log.write("", false);
		
		if(!Main.getClientManager().isLoggedIn(address)) {
			try {
				session.getRemote().sendString("notloggedin");
			} catch (IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Handler(80): " + e.getMessage(), true);
				Log.write("", false);
			}
		}else {
			try {
				session.getRemote().sendString("loggedin");
			}catch(IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Handler(88): " + e.getMessage(), true);
				Log.write("", false);
			}
		}
	}
	
	@OnWebSocketMessage
	public void onMessage(Session session, String message) {
		InetAddress address = session.getRemoteAddress().getAddress();
		Command.checkCommand(message, address);
	}
	
}
