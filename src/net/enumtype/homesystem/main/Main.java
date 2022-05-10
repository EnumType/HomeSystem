package net.enumtype.homesystem.main;

import java.io.IOException;
import java.util.Scanner;

import net.enumtype.homesystem.monitoring.Monitoring;
import net.enumtype.homesystem.server.ClientManager;
import net.enumtype.homesystem.websocket.WebSocket;
import net.enumtype.homesystem.xmlrpc.RoomManager;
import net.enumtype.homesystem.server.Commands;
import net.enumtype.homesystem.utils.StaticAI;
import net.enumtype.homesystem.utils.Data;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import net.enumtype.homesystem.xmlrpc.Rooms;
import net.enumtype.homesystem.xmlrpc.XmlRpcServer;

// Created by JavaException

public class Main {

	public static boolean scan;
	public static boolean reload;
	public static Scanner scanner;
	private static ClientManager clientManager;
	private static RoomManager roomManager;
	private static Monitoring monitoring;
	
	public static void main(String[] args) {
		try {
			Data.version = "v1.0.9-Beta-4";
			Data.saveAIData = true;
			Data.doAIPrediction = false; //TODO: Renew AI and test it. After this back to "true"!!
			Log.initLog();
			Log.write("loading libraries, please wait...", true);

			clientManager = new ClientManager();
			roomManager = new RoomManager();
			monitoring = new Monitoring();

			clientManager.load();
			roomManager.load();

			XmlRpcServer.getConfigs();
			StaticAI.checkAIData();
			StaticAI.startSavingData(Data.aiInterval);
			StaticAI.startPredictions(Data.aiInterval);
			StaticAI.startAutoTraining();
			Methods.startVersionChecking();
			WebSocket.startWebSocket(Data.wsport, Data.wssport, Data.resourcesDir, Data.resourcesDir + "//" + Data.wsKeystore, Data.wsKeystorePassword);
			startScanning();
		} catch (IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Main(51): " + e.getMessage(), false);
		}
		
	}
	
	public static void startScanning() {
		if(!scan) {
			scan = true;
			
			Thread thread = new Thread(() -> {
				reload = false;
				while(scan) {
					System.out.print(">");
					scanner = new Scanner(System.in);
					String command = scanner.nextLine().replaceAll(">", "");

					try {
						checkConsoleCommand(command);
					}catch(Exception ex) {
						ex.printStackTrace();
					}
				}
			});
			
			thread.start();
		}
	}
	
	public static void checkConsoleCommand(String command) throws Exception {
		if(command.equalsIgnoreCase("help")) {
			Commands.executeHelpCommand();
		}else if(command.equalsIgnoreCase("stop")) {
			scanner.close();
			Commands.executeStopCommand();
		}else if(command.startsWith("version")) {
			String[] args = command.split(" ");
			if(args.length == 2) {
				Commands.updateVersion(args[1]);
			}else {
				System.out.println("Usage: /version <version>");
			}
		}else if(command.startsWith("addperm")) {
			command = command.replaceFirst("addperm ", "");
			String[] args = command.split(" ");
			
			if(args.length == 2) {
				clientManager.addPermission(args[0], args[1]);
				Log.write(Methods.createPrefix() + "Added " + args[1] + " to user " + args[0], false);
			}else System.out.println("Usage: /addperm <Username> <Permission>");
		}else if(command.startsWith("removeperm")) {
			command = command.replaceFirst("removeperm ", "");
			final String[] args = command.split(" ");

			if(args.length == 2) {
				if(clientManager.removePermission(args[0], args[1])) {
					Log.write(Methods.createPrefix() + "Removed '" + args[1] + "' from user '" + args[0] + "'.", false);
				}else Log.write(Methods.createPrefix() + "Cannot remove '" + args[1] + "' from user '" + args[0] + "'.", false);
			}else System.out.println("Usage: /removeperm <Username> <Permission>");
		}else if(command.startsWith("reload")) {
			try {
				scan = false;
				reload = true;
				Data.saveAIData = false;
				Data.doAIPrediction = false;
				Data.saveAIData = true;
				Data.doAIPrediction = false; //TODO: Renew AI and test it. After this back to "true"!!
				clientManager.loadUserData();
				clientManager.loadUserPerm();
				XmlRpcServer.getConfigs();
				Rooms.loadData();
				StaticAI.startSavingData(Data.aiInterval);
				StaticAI.startPredictions(Data.aiInterval);
			}catch(IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Main(129): " + e.getMessage(), false);
			}
		}else if(command.startsWith("extract website")) {
			Methods.extractWebsite();
		}else if(command.startsWith("train now")) {
			StaticAI.trainNow();
		}
	}

	public static ClientManager getClientManager() {return clientManager;}
	public static RoomManager getRoomManager() {return roomManager;}
	public static Monitoring getMonitoring() {return monitoring;}
	
}