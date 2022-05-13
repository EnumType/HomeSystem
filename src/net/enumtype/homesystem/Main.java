package net.enumtype.homesystem;

import java.io.IOException;
import java.util.Scanner;

import net.enumtype.homesystem.rooms.AIManager;
import net.enumtype.homesystem.server.Command;
import net.enumtype.homesystem.server.WebSocketServer;
import net.enumtype.homesystem.utils.*;
import net.enumtype.homesystem.server.ClientManager;
import net.enumtype.homesystem.rooms.RoomManager;

// Created by EnumType

public class Main {

	private static final String commandPrefix = "> ";
	private static Thread scanningThread;
	private static ClientManager clientManager;
	private static RoomManager roomManager;
	private static AIManager aiManager;
	private static Monitoring monitoring;
	private static WebSocketServer wsServer;
	private static Log log;
	private static Data data;
	
	public static void main(String[] args) {
		try {
			log = new Log();
			data = new Data();
			data.setVersion("v1.0.9-Beta-10");

			aiManager = new AIManager();
			clientManager = new ClientManager();
			roomManager = new RoomManager();
			monitoring = new Monitoring();
			wsServer = new WebSocketServer();

			aiManager.startDataSaving(data.getAiInterval());
			aiManager.startPredictions(data.getAiInterval());
			aiManager.startAutoTraining();

			//Methods.startVersionChecking(); TODO: turn back on
			wsServer.start();
			startScanning();

			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				try {
					scanningThread.interrupt();
					clientManager.writeUserPerm(true);
					aiManager.stopDataSaving();
					aiManager.stopPredictions();
					aiManager.stopAutoTraining();
					aiManager.saveData();
					wsServer.stop();
				}catch(Exception e) {
					log.writeError(e);
				}
			}));
		}catch (IOException e) {
			log.writeError(e);
		}
		
	}
	
	public static void startScanning() {
		scanningThread = new Thread(() -> {
			while(true) {
				final Scanner scanner = new Scanner(System.in);
				final String command = scanner.nextLine().replaceAll(commandPrefix, "").toLowerCase();
				if(!command.startsWith("adduser")) log.write(commandPrefix + command, false);
				try {
					if(!command.isEmpty()) {
						Command.check("console" + command, null);
					}else System.out.print(commandPrefix);
				}catch(Exception e) {
					log.writeError(e);
				}
			}
		});

		scanningThread.start();
	}

	public static ClientManager getClientManager() {return clientManager;}
	public static RoomManager getRoomManager() {return roomManager;}
	public static AIManager getAiManager() {return aiManager;}
	public static Monitoring getMonitoring() {return monitoring;}
	public static Log getLog() {return log;}
	public static Data getData() {return data;}
	public static Thread getScanningThread() {return scanningThread;}
	public static String getCommandPrefix() {return commandPrefix;}
	
}