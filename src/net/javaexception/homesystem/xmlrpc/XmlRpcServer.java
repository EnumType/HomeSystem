package net.javaexception.homesystem.xmlrpc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.xmlrpc.XmlRpcException;
import org.apache.xmlrpc.client.XmlRpcClient;
import org.apache.xmlrpc.client.XmlRpcClientConfigImpl;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.server.Server;
import net.javaexception.homesystem.utils.Data;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

public class XmlRpcServer {
	
	private static HashMap<String, Object> cachedStates = new HashMap<String, Object>();
	
	public static void getConfigs() {
		try {
			
			File config = new File("config.txt");
			
			if(config.exists()) {
				Log.write(Methods.createPrefix() + "Getting Configs...", true);
				BufferedReader reader = new BufferedReader(new FileReader(config));
				String line;
				
				while((line = reader.readLine()) != null) {
					if(line.startsWith("XmlRpc-Address")) {
						line = line.replace("XmlRpc-Address: ", "");
						Data.xmlrpcAddress = line;
					}else if(line.startsWith("XmlRpc-Port")) {
						line = line.replace("XmlRpc-Port: ", "");
						try {
							int port = Integer.parseInt(line);
							Data.xmlrpcPort = port;
						}catch(NumberFormatException e) {
							e.printStackTrace();
							Log.write(Methods.createPrefix() + "Error in XmlRpcServer(52): " + e.getMessage(), false);
						}
					}else if(line.startsWith("HmIP-Port")) {
						line = line.replace("HmIP-Port: ", "");
						try {
							int port = Integer.parseInt(line);
							Data.hmipPort = port;
						}catch(NumberFormatException e) {
							e.printStackTrace();
							Log.write(Methods.createPrefix() + "Error in XmlRpcServer(61): " + e.getMessage(), false);
						}
					}else if(line.startsWith("Server-Port")) {
						line = line.replace("Server-Port: ", "");
						try {
							int port = Integer.parseInt(line);
							Data.serverPort = port;
						}catch(NumberFormatException e) {
							e.printStackTrace();
							Log.write(Methods.createPrefix() + "Error in XmlRpcServer(70): " + e.getMessage(), false);
						}
					}else if(line.startsWith("Resources-Dir")) {
						line = line.replace("Resources-Dir: ", "");
						Data.resourcesDir = line;
					}else if(line.startsWith("WebSocket-Http") && !line.startsWith("WebSocket-Https")) {
						line = line.replace("WebSocket-Http: ", "");
						try {
							int port = Integer.parseInt(line);
							Data.wsport = port;
						}catch(NumberFormatException e) {
							e.printStackTrace();
							Log.write(Methods.createPrefix() + "Error in XmlRpcServer(82): " + e.getMessage(), false);
						}
					}else if(line.startsWith("WebSocket-Https")) {
						line = line.replace("WebSocket-Https: ", "");
						try {
							int port = Integer.parseInt(line);
							Data.wssport = port;
						}catch(NumberFormatException e) {
							e.printStackTrace();
							Log.write(Methods.createPrefix() + "Error in XmlRpcServer(91): " + e.getMessage(), false);
						}
					}else if(line.startsWith("WebSocket-Keystore") && !line.startsWith("WebSocket-KeystorePassword")) {
						line = line.replace("WebSocket-Keystore: ", "");
						Data.wsKeystore = line;
					}else if(line.startsWith("WebSocket-KeystorePassword")) {
						line = line.replace("WebSocket-KeystorePassword: ", "");
						Data.wsKeystorePassword = line;
					}else if(line.startsWith("BrightnessSensor") && !line.startsWith("BrightnessSensorHmIP")) {
						line = line.replace("BrightnessSensor: ", "");
						Data.aiBright = line;
					}else if(line.startsWith("BrightnessSensorHmIP")) {
						line = line.replace("BrightnessSensorHmIP: ", "");
						Data.aiBrightHmIP = Boolean.parseBoolean(line);
					}else if(line.startsWith("AI-Interval")) {
						line = line.replace("AI-Interval: ", "");
						Data.aiInterval = Integer.parseInt(line);
					}
				}
				
				reader.close();
			}else {
				Log.write(Methods.createPrefix() + "Creating Configs...", true);
				InputStream stream = Main.class.getResourceAsStream("/config.txt");
				BufferedReader in = new BufferedReader(new InputStreamReader(stream));
				BufferedWriter out = new BufferedWriter(new FileWriter(config));
				
				String line;
				while((line = in.readLine()) != null) {
					out.write(line + "\r\n");
				}
				
				in.close();
				out.close();
				getConfigs();
			}
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in XmlRpcServer(129): " + e.getMessage(), false);
		}
	}
	
	public static void setValue(String id, String value_key, Object value, InetAddress address, boolean hmip, String room) {
		new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
					XmlRpcClient client = new XmlRpcClient();
					
					if(hmip) {
						config.setServerURL(new URL("http://" + Data.xmlrpcAddress + ":" + Data.hmipPort));
					}else {
						config.setServerURL(new URL("http://" + Data.xmlrpcAddress + ":" + Data.xmlrpcPort));
					}
					
					cacheRoomStates(room);
					
					client.setConfig(config);
					
					if(getValue(id, value_key, hmip) != value) {
						client.execute("setValue", new Object[]{id, value_key, value});
					}
					
					if(address != null) {
						Server.sendCommand(address, "ok");
					}
				}catch(XmlRpcException | IOException e) {
					e.printStackTrace();
					Log.write(Methods.createPrefix() + "Error in XmlRpcServer(160): " + e.getMessage(), false);
					try {
						if(address != null) {
							Server.sendCommand(address, "failure");
						}
					} catch (IOException e1) {
						e1.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in XmlRpcServer(167): " + e.getMessage(), false);
					}
				}
				
				removeCachedRoom(room);
			}
		}).start();
	}
	
	public static Object getValue(String id, String value, boolean hmip) {		
		try {
			XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
			XmlRpcClient client = new XmlRpcClient();
			
			if(hmip) {
				config.setServerURL(new URL("http://" + Data.xmlrpcAddress + ":" + Data.hmipPort));
			}else {
				config.setServerURL(new URL("http://" + Data.xmlrpcAddress + ":" + Data.xmlrpcPort));
			}
			
			if(cachedStates.containsKey(id)) {
				return cachedStates.get(id); 
			}
			
			client.setConfig(config);
			
			return client.execute("getValue", new Object[]{id, value});
		}catch(MalformedURLException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in XmlRpcServer(196): " + e.getMessage(), false);
		}catch(XmlRpcException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in XmlRpcServer(199): " + e.getMessage(), false);
		}
		return "";
	}
	
	public static ArrayList<String> states(String room, String device, boolean hmip) {
		ArrayList<String> states = new ArrayList<String>();
		String address = Rooms.getDeviceAddress(room, device);
		
		if(Rooms.getDeviceType(room, device).equalsIgnoreCase("ROLL")) {
			states.add("LEVEL:" + getValue(address, "LEVEL", hmip).toString());
			states.add("WORKING:" + getValue(address, "WORKING", hmip).toString());
		}else if(Rooms.getDeviceType(room, device).equalsIgnoreCase("LAMP")) {
			states.add("STATE:" + getValue(address, "STATE", hmip).toString());
		}
		
		return states;
	}
	
	public static String formatStateResult(String result) {
		String firstChar = Character.toString(result.charAt(0));
		String state = result.replace(".", "");
		
		if(firstChar.startsWith("0")) {
			state = state.replaceFirst("0", "");
		}else if(firstChar.startsWith("1")) {
			state = state + "0";
		}
		return state;
	}
	
	public static void cacheRoomStates(String room) {		
		for(String device : Rooms.getRoomDevices(room)) {
			String id = Rooms.getDeviceAddress(room, device);
			
			if(!cachedStates.containsKey(id)) {
				if(Rooms.getDeviceType(room, device).equalsIgnoreCase("ROLL")) {
					cachedStates.put(id, getValue(Rooms.getDeviceAddress(room, device), "LEVEL", Rooms.getDeviceHmIP(room, device)));
					cachedStates.put(id, getValue(Rooms.getDeviceAddress(room, device), "WORKING", Rooms.getDeviceHmIP(room, device)));
				}else if(Rooms.getDeviceType(room, device).equalsIgnoreCase("LAMP")) {
					cachedStates.put(id, getValue(Rooms.getDeviceAddress(room, device), "STATE", Rooms.getDeviceHmIP(room, device)));
				}
			}
		}
	}
	
	public static void removeCachedRoom(String room) {
		for(String device : Rooms.getRoomDevices(room)) {
			String id = Rooms.getDeviceAddress(room, device);
			
			if(cachedStates.containsKey(id)) {
				cachedStates.remove(id);
			}
		}
	}
	
	public static void clearCache() {
		if(!cachedStates.isEmpty()) {
			cachedStates.clear();
		}
	}
	
}
