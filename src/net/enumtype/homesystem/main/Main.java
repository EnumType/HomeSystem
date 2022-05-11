package net.enumtype.homesystem.main;

import java.io.IOException;
import java.util.Scanner;

import net.enumtype.homesystem.utils.*;
import net.enumtype.homesystem.server.ClientManager;
import net.enumtype.homesystem.utils.WebSocketServer;
import net.enumtype.homesystem.xmlrpc.RoomManager;
import net.enumtype.homesystem.server.Commands;

// Created by EnumType

public class Main {

	private static Scanner scanner;
	private static ClientManager clientManager;
	private static RoomManager roomManager;
	private static AIManager aiManager;
	private static Monitoring monitoring;
	private static WebSocketServer wsServer;
	private static Log log;
	private static Data data;
	private static boolean scan = true;
	
	public static void main(String[] args) {
		try {
			log = new Log();
			data = new Data();
			data.setVersion("v1.0.9-Beta-5");
			log.write("loading libraries, please wait...", true);

			clientManager = new ClientManager();
			roomManager = new RoomManager();
			aiManager = new AIManager();
			monitoring = new Monitoring();
			wsServer = new WebSocketServer(data.getWsPort(), data.getWssPort(), data.getResourcesDir(),
									data.getResourcesDir() + "//" + data.getWsKeystore(), data.getWsKeystorePassword());

			aiManager.startDataSaving(data.getAiInterval());
			aiManager.startPredictions(data.getAiInterval());
			aiManager.startAutoTraining();

			Methods.startVersionChecking();
			wsServer.start();
			startScanning();
		} catch (IOException e) {
			e.printStackTrace();
			log.write(Methods.createPrefix() + "Error in Main(51): " + e.getMessage(), false);
		}
		
	}
	
	public static void startScanning() {
		Thread thread = new Thread(() -> {
			while(scan) {
				System.out.print(">");
				scanner = new Scanner(System.in);
				if(!scan) break;

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
	
	public static void checkConsoleCommand(String command) throws Exception {
		if(command.equalsIgnoreCase("help")) {
			Commands.executeHelpCommand();
		}else if(command.equalsIgnoreCase("stop")) {
			scanner.close();
			wsServer.stop();
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
				log.write(Methods.createPrefix() + "Added " + args[1] + " to user " + args[0], false);
			}else System.out.println("Usage: /addperm <Username> <Permission>");
		}else if(command.startsWith("removeperm")) {
			command = command.replaceFirst("removeperm ", "");
			final String[] args = command.split(" ");

			if(args.length == 2) {
				if(clientManager.removePermission(args[0], args[1])) {
					log.write(Methods.createPrefix() + "Removed '" + args[1] + "' from user '" + args[0] + "'.", false);
				}else log.write(Methods.createPrefix() + "Cannot remove '" + args[1] + "' from user '" + args[0] + "'.", false);
			}else System.out.println("Usage: /removeperm <Username> <Permission>");
		}else if(command.startsWith("reload")) {
			try {
				scan = false;
				aiManager.stopDataSaving();
				aiManager.stopPredictions();
				aiManager.stopAutoTraining();
				clientManager.loadUserData();
				clientManager.loadUserPerm();
				data.load();
				roomManager.load();
				aiManager.startDataSaving(data.getAiInterval());
				aiManager.startPredictions(data.getAiInterval());
				aiManager.startAutoTraining();
				startScanning();
			}catch(IOException e) {
				e.printStackTrace();
				log.write(Methods.createPrefix() + "Error in Main(129): " + e.getMessage(), false);
			}
		}else if(command.startsWith("extract website")) {
			Methods.extractWebsite();
		}else if(command.startsWith("train now")) {
			aiManager.trainAll();
		}
	}

	public static ClientManager getClientManager() {return clientManager;}
	public static RoomManager getRoomManager() {return roomManager;}
	public static Monitoring getMonitoring() {return monitoring;}
	public static Log getLog() {return log;}
	public static Data getData() {return data;}
	
}