package net.javaexception.homesystem.websocket;

import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;

import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketClose;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketConnect;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketError;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketMessage;
import org.eclipse.jetty.websocket.api.annotations.WebSocket;

import net.javaexception.homesystem.server.Client;
import net.javaexception.homesystem.server.Command;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

@WebSocket(maxIdleTime=120000)
public class Handler{
	
	public static HashMap<InetAddress, Session> sessions = new HashMap<InetAddress, Session>();
	public static ArrayList<InetAddress> changingConnection = new ArrayList<InetAddress>();
	
	@OnWebSocketClose
	public void onClose(Session session, int statusCode, String reason) {
		InetAddress address = session.getRemoteAddress().getAddress();
		if(!changingConnection.contains(address)) {
			if(sessions.containsKey(address)) {
				sessions.remove(address);
			}
			
			if(Client.isLoggedIn(address)) {
				Client.logoutClient(Client.getUsername(address), address);
			}
		}else {
			sessions.remove(address);
		}
	}
	
	@OnWebSocketError
	public void onError(Session session, Throwable t) {
		InetAddress address = session.getRemoteAddress().getAddress();
		if(session != null) {
			if(sessions.containsKey(address)) {
				sessions.remove(address);
			}
		}
		
		if(Client.isLoggedIn(address)) {
			Client.logoutClient(Client.getUsername(address), address);
		}
	}
	
	@OnWebSocketConnect
	public void onConnect(Session session) {
		InetAddress address = session.getRemoteAddress().getAddress();
		
		if(!sessions.containsKey(address)) {
			sessions.put(address, session);
		}else {
			sessions.remove(address);
			sessions.put(address, session);
		}
		
		if(!changingConnection.contains(address)) {
			Log.write(Methods.createPrefix() + "New Connection to WebSocket Server:", true);
			Log.write("INETADDRESS: " + address, true);
			Log.write("", false);
		}else {
			changingConnection.remove(address);
		}
		
		if(!Client.isLoggedIn(address)) {
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
		
		if(!message.equalsIgnoreCase("changeconnection")) {
			Command.checkCommand(message, address);
		}else {
			changingConnection.add(address);
			session.close();
		}
	}
	
}
