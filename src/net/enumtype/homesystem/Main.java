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
			data.setVersion("v1.0.9-Beta-8");
			log.write("loading libraries, please wait...", true, false);

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

			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				try {
					clientManager.writeUserPerm(true);
					scanningThread.interrupt();
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
				System.out.print(">");
				final Scanner scanner = new Scanner(System.in);
				final String command = scanner.nextLine().replaceAll(">", "");
				try {
					Command.check(command.toLowerCase(), null);
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
	public static WebSocketServer getWsServer() {return wsServer;}
	public static Thread getScanningThread() {return scanningThread;}
	
}