package net.javaexception.homesystem.server;

import java.io.IOException;
import java.net.InetAddress;
import java.security.PublicKey;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import net.javaexception.homesystem.utils.Data;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;
import net.javaexception.homesystem.websocket.WebSocket;

public class Client {
	
	public static HashMap<String, String> users = new HashMap<String, String>();
	public static HashMap<String, ArrayList<String>> permissions = new HashMap<>();
	private static HashMap<InetAddress, PublicKey> userKeys = new HashMap<InetAddress, PublicKey>();
	private static HashMap<InetAddress, String> clients = new HashMap<InetAddress, String>();
	private static HashMap<String, InetAddress> loggedInUsers = new HashMap<String, InetAddress>();
	private static HashMap<String, String> userID = new HashMap<String, String>();
	private static final String numbers = "0123456789";
	private static final String characters = "abcdefghijklmnopqrstuvwxyz";
	private static final char[] alphabet = (numbers + characters).toCharArray();
	
	public static String createToken() {
		char[] randomString = new char[64];
		Random random = new Random();
		
		for(int i = 0; i < randomString.length; i++) {
			randomString[i] = alphabet[random.nextInt(alphabet.length)];
		}
		
		boolean containsDigits = false;
		boolean containsCharacters = false;
		for(Character c: randomString) {
			containsDigits |= numbers.contains(c.toString());
			containsCharacters |= characters.contains(c.toString());
		}
		
		if(64 < 2 || (containsDigits && containsCharacters)) {
			return new String(randomString);
		}else {
			return createToken();
		}
	}
	
	public static void verifyClient(InetAddress address) {
		if(!clients.containsKey(address)) {
			String token = createToken();
			clients.put(address, token);
			Log.write(Methods.createPrefix() + "Verified Client '" + token + "' with InetAddress " + address, false);
		}else {
			Log.write(Methods.createPrefix() + "Client '" + clients.get(address)
			+ "' is allready registered with InteAddress " + address, false);
		}
	}
	
	public static void loginClient(String user, String pass, InetAddress address) {
		if(users.containsKey(user)) {
			if(users.get(user).equalsIgnoreCase(pass)) {
				verifyClient(address);
				loggedInUsers.put(user, address);
				userID.put(user, getClientID(address));
				Log.write(Methods.createPrefix() + "Client logged in:", true);
				Log.write("USERNAME: " + user, true);
				Log.write("INETADDRESS: " + address, true);
				Log.write("CLIENTID: " + getClientID(address), true);
				Log.write("", false);
				
  				try {
  					Server.sendCommand(address, "verifylogin " + user);
  					if(Data.newVersion) {
  						Thread.sleep(200);
  						Server.sendCommand(address, "update " + Data.version);
  					}
  				}catch (IOException | InterruptedException e) {
  					e.printStackTrace();
  					Log.write(Methods.createPrefix() + "Error in Client(81): " + e.getMessage(), false);
  				}
			}else {
				Log.write(Methods.createPrefix() + "Client with InetAddress " + address + " tried to login:", true);
				Log.write("USERNAME: " + user, true);
				Log.write("", false);
				
				try {
					Server.sendCommand(address, "wrong data");
				} catch (IOException e) {
					e.printStackTrace();
  					Log.write(Methods.createPrefix() + "Error in Client(92): " + e.getMessage(), false);
				}
			}
		}else {
			Log.write(Methods.createPrefix() + "Client with InetAddress " + address + " tried to login:", true);
			Log.write("USERNAME: " + user, true);
			Log.write("", false);
			
			try {
				Server.sendCommand(address, "wrong data");
			} catch (IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Client(104): " + e.getMessage(), false);
			}
		}
	}
	
	public static void logoutClient(String user, InetAddress address) {
		if(loggedInUsers.containsKey(user)) {
			loggedInUsers.remove(user);
			Log.write(Methods.createPrefix() + "Client '" + getClientID(address) + "' logged out!", true);
			Log.write("USERNAME: " + user, true);
			Log.write("INETADDRESS: " + address, true);
			Log.write("CLIENTID: " + getClientID(address), true);
			Log.write("", false);
			
			if(WebSocket.isWebSocket(address)) {
				WebSocket.closeConnection(address);
			}
		}
	}
	
	public static String getClientID(InetAddress address) {
		return clients.get(address);
	}
	
	public static String getPassword(String username) {
		return users.get(username);
	}
	
	public static boolean isLoggedIn(InetAddress address) {
		return loggedInUsers.containsValue(address);
	}
	
	public static String getOnlineUsers() {
		String onlineUsers = "";
		
		for(String s : loggedInUsers.keySet()) {
			onlineUsers = onlineUsers + " " + s;
		}
		
		return onlineUsers;
	}
	
	public static void addPermission(String user, String permission) {
		if(!permissions.containsKey(user)) {
			ArrayList<String> list = new ArrayList<>();
			list.add(permission);
			permissions.put(user, list);
		}else {
			permissions.get(user).add(permission);
		}
	}
	
	public static void removePermission(String user, String permission) {
		if(permissions.containsKey(user)) {
			permissions.get(user).remove(permission);
			Log.write(Methods.createPrefix() + "Removed " + permission + " from user " + user, false);
		}else {
			Log.write(Methods.createPrefix() + "There is no user named " + user, false);
		}
	}
	
	public static boolean hasPermission(String user, String permission) {
		if(permissions.get(user) != null) {
			return permissions.get(user).contains(permission) || permissions.get(user).contains("*");
		}
		return false;
	}
	
	public static List<String> getPermissions(String user) {
		return permissions.get(user);
	}
	
	public static String getUsername(InetAddress address) {
		String user = "";
		
		for(String users : loggedInUsers.keySet()) {
			if(loggedInUsers.get(users).equals(address)) {
				user = users;
			}
		}
		
		return user;
	}
	
	public static void addUserKey(InetAddress address, PublicKey key) {
		if(!userKeys.containsKey(address)) {
			userKeys.put(address, key);
		}
	}
	
	public static void removeUserKey(InetAddress address) {
		if(userKeys.containsKey(address)) {
			userKeys.remove(address);
		}
	}
	
	public static PublicKey getUserKey(InetAddress address) {
		return userKeys.get(address);
	}
	
	public static boolean userKeyExists(InetAddress address) {
		return userKeys.get(address) != null;
	}
	
	public static void clearCacheData() {
		Log.write(Methods.createPrefix() + "Clearing User Cache...", true);
		
		users.clear();
		permissions.clear();
		userKeys.clear();
		clients.clear();
		loggedInUsers.clear();
		userID.clear();
	}
	
}
