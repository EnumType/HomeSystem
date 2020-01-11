package net.javaexception.homesystem.main;

import java.io.IOException;
import java.util.Scanner;

import net.javaexception.homesystem.server.Client;
import net.javaexception.homesystem.server.Commands;
import net.javaexception.homesystem.server.Server;
import net.javaexception.homesystem.utils.AI;
import net.javaexception.homesystem.utils.Crypto;
import net.javaexception.homesystem.utils.Data;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;
import net.javaexception.homesystem.utils.UserData;
import net.javaexception.homesystem.websocket.WebSocket;
import net.javaexception.homesystem.xmlrpc.Rooms;
import net.javaexception.homesystem.xmlrpc.XmlRpcServer;

// Created by JavaException

public class Main {
	
	public static Crypto crypto;
	public static boolean scan;
	public static boolean reload;
	public static Scanner scanner;
	
	public static void main(String[] args) {
		Data.isWorking = true;
		Data.saveAIData = true;
		Data.doAIPrediction = true;
		Log.initLog();
		Log.write("loading libaries, please wait...", true);
		try {
			crypto = new Crypto("RSA", 2048);
			crypto.generate();
			
			UserData.loadUserData();
			UserData.loadUserPerm();
			XmlRpcServer.getConfigs();
			Rooms.loadData();
			AI.checkAIData();
			AI.startSavingData(Data.aiInterval);
			AI.startPredictions(Data.aiInterval);
			AI.startAutoTraining();
			Methods.startVersionChecking();
			WebSocket.startWebSocket(Data.wsport, Data.wssport, Data.resourcesDir, Data.resourcesDir + "//" + Data.wsKeystore, Data.wsKeystorePassword);
			Server.start();
		} catch (IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Main(51): " + e.getMessage(), false);
		}
		
	}
	
	public static void startScanning() {		
		if(!scan) {
			scan = true;
			
			Thread thread = new Thread(new Runnable() {
				@Override
				public void run() {
					reload = false;
					while(scan) {
						System.out.print(">");
						Data.isWorking = false;
						scanner = new Scanner(System.in);
						String command = scanner.nextLine().replaceAll(">", "");
						
						checkConsoleCommand(command);
					}
					if(reload) {
						try {
							Server.start();
						} catch (IOException e) {
							e.printStackTrace();
							Log.write(Methods.createPrefix() + "Error in Main(77): " + e.getMessage(), false);
						}
					}
				}
			});
			
			thread.start();
		}
	}
	
	public static void checkConsoleCommand(String command) {
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
				Client.addPermission(args[0], args[1]);
				Log.write(Methods.createPrefix() + "Added " + args[1] + " to user " + args[0], true);
				Log.write("", false);
			}else {
				System.out.println("Usage: /addperm <Username> <Permission>");
			}
		}else if(command.startsWith("reload")) {
			try {
				scan = false;
				reload = true;
				Data.saveAIData = false;
				Data.doAIPrediction = false;
				Server.stop();
				Client.clearCacheData();
				Data.saveAIData = true;
				Data.doAIPrediction = true;
				UserData.loadUserData();
				UserData.loadUserPerm();
				XmlRpcServer.getConfigs();
				Rooms.loadData();
				AI.startSavingData(Data.aiInterval);
				AI.startPredictions(Data.aiInterval);
			}catch(IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Main(129): " + e.getMessage(), false);
			}
		}else if(command.startsWith("extract website")) {
			Methods.extractWebsite();
		}else if(command.startsWith("train now")) {
			AI.trainNow();
		}
	}
	
}