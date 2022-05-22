package net.enumtype.homesystem.server;

import java.io.IOException;
import java.util.Scanner;

import net.enumtype.homesystem.plugin.PluginManager;
import net.enumtype.homesystem.server.utils.*;

// Created by EnumType

public class HomeSystem {

	private static final String commandPrefix = "> ";
	private static Thread scanningThread;
	private static ClientManager clientManager;
	private static RoomManager roomManager;
	private static AIManager aiManager;
	private static PluginManager pluginManager;
	private static Monitoring monitoring;
	private static WebSocketServer wsServer;
	private static Data data;
	private static Logger logger;
	
	public static void main(String[] args) {
		try {
			logger = new Logger(System.out);
			data = new Data();
			data.setVersion("v1.0.9-Beta-14");

			aiManager = new AIManager();
			clientManager = new ClientManager();
			roomManager = new RoomManager();
			pluginManager = new PluginManager();
			monitoring = new Monitoring();
			wsServer = new WebSocketServer();

			aiManager.trainAll();
			aiManager.startDataSaving(data.getAiInterval());
			aiManager.startPredictions(data.getAiInterval());
			aiManager.startAutoTraining();

			//Methods.startVersionChecking(); TODO: turn back on
			wsServer.start();
			startScanning();

			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				try {
					pluginManager.unloadPlugins();
					scanningThread.interrupt();
					clientManager.writeUserPerm(true);
					aiManager.stopDataSaving();
					aiManager.stopPredictions();
					aiManager.stopAutoTraining();
					aiManager.interruptAll();
					aiManager.saveData();
					wsServer.stop();
				}catch(Exception e) {
					e.printStackTrace();
				}
			}));
		}catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void startScanning() {
		scanningThread = new Thread(() -> {
			while(true) {
				final Scanner scanner = new Scanner(System.in);
				final String command = scanner.nextLine().replaceAll(commandPrefix, "").toLowerCase();
				logger.setCurrentLine("");
				try {
					if(!command.isEmpty()) {
						Command.check("console" + command, null);
					}else System.out.print(commandPrefix);
				}catch(Exception e) {
					e.printStackTrace();
				}
			}
		});

		scanningThread.start();
	}

	public static ClientManager getClientManager() {return clientManager;}
	public static RoomManager getRoomManager() {return roomManager;}
	public static AIManager getAiManager() {return aiManager;}
	public static PluginManager getPluginManager() {return pluginManager;}
	public static Monitoring getMonitoring() {return monitoring;}
	public static Data getData() {return data;}
	public static Thread getScanningThread() {return scanningThread;}
	public static String getCommandPrefix() {return commandPrefix;}
	
}