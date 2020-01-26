package net.javaexception.homesystem.server;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.ConnectException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.utils.Crypto;
import net.javaexception.homesystem.utils.Data;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;
import net.javaexception.homesystem.websocket.WebSocket;

public class Server {
	
	private static ServerSocket serverSocket;
	private static HashMap<InetAddress, Socket> sockets;
	
	public static void start() throws IOException {
		Data.version = "v1.0.8";
		
		Log.write(Methods.createPrefix() + "Starting Server version: " + Data.version + " Port: " + Data.serverPort, true);
		System.out.println("");
		
		if(sockets != null) {
			if(!sockets.isEmpty()) {
				sockets.clear();
			}
		}
		
		sockets = new HashMap<InetAddress, Socket>();
		Crypto crypto = Main.crypto;
		Data.isServerStarted = true;
		Main.startScanning();
		
		serverSocket = new ServerSocket(Data.serverPort);
		
		while(Data.isServerStarted) {
			String line = "";
			Socket socket = serverSocket.accept();
			
			if(!sockets.containsKey(socket.getInetAddress())) {
				sockets.put(socket.getInetAddress(), socket);
			}else {
				sockets.remove(socket.getInetAddress());
				sockets.put(socket.getInetAddress(), socket);
			}
			
			BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
			line = in.readLine();
			
			if(!line.startsWith("clientkey")) {
				if(Client.userKeyExists(socket.getInetAddress())) {
					line = new String(crypto.decrypt(Base64.getDecoder().decode(line.getBytes()), crypto.getPrivateKey()));
				}
			}
			
			final String command = line;
			Thread thread = new Thread(new Runnable() {
				@Override
				public void run() {
					Command.checkCommand(command, socket.getInetAddress());
				}
			});
			thread.start();
		}
	}
	
	public static void sendCommand(InetAddress address, String command) throws IOException, UnknownHostException {
		if(WebSocket.isWebSocket(address)) {
			WebSocket.sendCommand(command, address);
		}else if(isSocket(address)){
			try {
				Socket socket = getAddressSocket(address);
				PrintStream out = new PrintStream(socket.getOutputStream());
				if(Client.userKeyExists(address)) {
					Crypto crypto = Main.crypto;
					out.println(Base64.getEncoder().encodeToString(crypto.encrypt(command, Client.getUserKey(address))));
				}else {
					out.println(command);
				}
			}catch(ConnectException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Server(93): " + e.getMessage(), false);
			}
		}
	}
	
	public static void sendPackageCommand(InetAddress address, ArrayList<String> commands, String type, Crypto crypto) throws IOException, UnknownHostException {
		if(WebSocket.isWebSocket(address)) {			
			for(int i = 0; i < commands.size(); i++) {
				WebSocket.sendCommand("type:" + type + " " + commands.get(i), address);
			}
		}else if(isSocket(address)){
			try {
				Socket socket = getAddressSocket(address);
				PrintStream out = new PrintStream(socket.getOutputStream());
				
				out.println(Base64.getEncoder().encodeToString(crypto.encrypt("lines: " + (commands.size() + 1), Client.getUserKey(address))));
				out.println(Base64.getEncoder().encodeToString(crypto.encrypt("type: " + type, Client.getUserKey(address))));
				
				for(int i = 0; i < commands.size(); i++) {
					out.println(Base64.getEncoder().encodeToString(crypto.encrypt(commands.get(i), Client.getUserKey(address))));
				}
			}catch(IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Server(116): " + e.getMessage(), false);
			}
		}
	}
	
	private static Socket getAddressSocket(InetAddress address) {
		return sockets.get(address);
	}
	
	private static boolean isSocket(InetAddress address) {
		return sockets.get(address) != null;
	}
	
	public static void stop() throws IOException {
		Data.isServerStarted = false;
		
		try {
			Socket socket = new Socket("localhost", Data.serverPort);
			PrintStream out = new PrintStream(socket.getOutputStream());
			out.println("connect");
			socket.close();
		}catch(ConnectException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Server(139): " + e.getMessage(), true);
		}
		
		serverSocket.close();
	}
	
}
