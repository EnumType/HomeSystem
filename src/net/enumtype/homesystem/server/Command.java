package net.enumtype.homesystem.server;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.enumtype.homesystem.server.exceptions.UnknownCommandException;
import net.enumtype.homesystem.server.utils.*;
import org.eclipse.jetty.websocket.api.Session;

public class Command {
	public static void check(String command, Session session) throws UnknownCommandException {
		if(command.startsWith("console") && session == null) ConsoleCommand.check(command.replace("console", ""));
		if(session == null) return;

		final ClientManager clientManager = HomeSystem.getClientManager();
		if(clientManager.isLoggedIn(session) || command.startsWith("login") || command.startsWith("connect")) {
			final Client client = !command.toLowerCase().startsWith("login") ?
					clientManager.getClient(session) : new Client(session, "", "");
			final String[] args = Arrays.copyOfRange(command.split(" "), 1, command.split(" ").length);
			command = command.split(" ")[0];
			HomeSystem.getPluginManager().triggerCommand(command, args);

			switch (command.toLowerCase()) {
				case "login":
					if(args.length == 2) {
						client.setLoginData(args[0], args[1]);
						if(client.login()) {
							client.sendMessage("verifylogin " + client.getName());
						}else client.sendMessage("wrongdata");
					}
					break;
				case "logout":
					client.logout();
					break;
				case "changeconnection":
					clientManager.getClient(session).changeConnection(true);
					break;
				case "xmlrpc":
					XmlRpcCommand.check(client, String.join(" ", args));
					break;
				case "getusername":
					client.sendMessage("user:" + client.getName());
					break;
				case "monitoring":
					if(!client.hasPermission("system.monitoring")) {
						client.sendMessage("noperm");
						break;
					}

					MonitoringCommand.check(String.join(" ", args), client);
					break;
				default:
					throw new UnknownCommandException(client, command);
			}

		}else {
			System.out.println("Client with InetAddress: " + session.getRemoteAddress().toString() +
							" tried to execute command: " + command);
			try {
				session.getRemote().sendString("notloggedin");
			}catch(IOException e) {
				e.printStackTrace();
			}
		}
	}
}

class XmlRpcCommand {
	public static void check(Client client, String command) throws UnknownCommandException {
		final RoomManager roomManager = HomeSystem.getRoomManager();
		final String[] args = Arrays.copyOfRange(command.split(" "), 1, command.split(" ").length);
		command = command.split(" ")[0];

		switch (command.toLowerCase()) {
			case "getrooms":
				final List<String> rooms = new ArrayList<>();

				roomManager.getRooms().forEach(room -> rooms.add(room.getName()));

				client.sendMessage("rooms", rooms);
				break;
			case "getdevices":
				if(args.length == 1) {
					String roomName = args[0];

					if(roomManager.existsRoom(roomName)) {
						final Room room = roomManager.getRoom(roomName);

						if(client.hasPermission(room.getPermission())) {
							client.sendMessage("roomdevices " + roomName, room.getDeviceNames());
						}else client.sendMessage("noperm " + roomName);
					}else client.sendMessage("noroom " + roomName);
				}else client.sendMessage("failure");
				break;
			case "getdevicetype":
				if(args.length == 2) {
					String roomName = args[0];
					String deviceName = args[1];

					if(roomManager.existsRoom(roomName)) {
						final Room room = roomManager.getRoom(roomName);

						if(client.hasPermission(room.getPermission())) {
							if(room.hasDevice(deviceName)) {
								client.sendMessage("device " + roomName + " " + deviceName + " " +
										room.getDevice(deviceName).getType());
							}else client.sendMessage("nodevice " + deviceName);
						}else client.sendMessage("noperm " + room);
					}else client.sendMessage("noroom " + roomName);
				}else client.sendMessage("failure");
				break;
			case "getdevicestate":
				if(args.length == 2) {
					final String roomName = args[0];
					final String deviceName = args[1];

					if(roomManager.existsRoom(roomName)) {
						final Room room = roomManager.getRoom(roomName);

						if(client.hasPermission(room.getPermission())) {
							if(room.hasDevice(deviceName)) {
								client.sendMessage("states " + deviceName, room.getDevice(deviceName).getStates());
							}else client.sendMessage("nodevice " + deviceName);
						}else client.sendMessage("noperm " + roomName);
					}else client.sendMessage("noroom " + roomName);
				}else client.sendMessage("failure");
				break;
			case "setdevice":
				if(args.length == 3) {
					String roomName = args[0];
					String deviceName = args[1];
					Object value = args[2].equalsIgnoreCase("false") || args[2].equalsIgnoreCase("true") ?
							Boolean.parseBoolean(args[2]) : args[2];

					if(roomManager.existsRoom(roomName)) {
						final Room room = roomManager.getRoom(roomName);

						if(client.hasPermission(room.getPermission())) {
							if(room.hasDevice(deviceName)) {
								final Device device = room.getDevice(deviceName);
								if(value.toString().equalsIgnoreCase("STOP")) {
									device.stop();
								}else device.setValue(value);

							}else client.sendMessage("nodevice " + deviceName);
						}else client.sendMessage("noperm " + roomName);
					}else client.sendMessage("noroom " + roomName);
				}else client.sendMessage("failure");
				break;
			default:
				throw new UnknownCommandException(client, command);
		}
	}
}

class MonitoringCommand {
	public static void check(String command, Client client) throws UnknownCommandException {
		final Monitoring monitoring = HomeSystem.getMonitoring();
		final String[] args = Arrays.copyOfRange(command.split(" "), 1, command.split(" ").length);
		command = command.split(" ")[0];

		switch (command.toLowerCase()) {
			case "isxmlrpcreachable":
				if(args.length == 1) {
					client.sendMessage("xmlrpcreachable " + monitoring.isXmlRpcReachable(Integer.parseInt(args[0])));
				}else client.sendMessage("failure");

				break;
			case "getlog":
				client.sendMessage("log", monitoring.getLog());
				break;
			default:
				throw new UnknownCommandException(client, command);
		}
	}
}

class ConsoleCommand {
	public static void check(String command) {
		final ClientManager clientManager = HomeSystem.getClientManager();
		final String[] args = Arrays.copyOfRange(command.split(" "), 1, command.split(" ").length);
		command = command.split(" ")[0];

		try {
			switch (command.toLowerCase()) {
				case "help":
					printHelp();
					break;
				case "stop":
					stopSystem();
					break;
				case "version":
					System.out.println("Current version of Home-System: " + HomeSystem.getData().getVersion());
					break;
				case "addperm":
					if(args.length != 2) {
						System.out.println("Usage: addperm <Username> <Permission>");
						break;
					}
					clientManager.addPermission(args[0], args[1]);
					System.out.println("Added permission '" + args[1] + "' to user '" + args[0] + "'.");
					break;
				case "removeperm":
					if(args.length == 2) {
						if(clientManager.removePermission(args[0], args[1])) {
							System.out.println("Removed '" + args[1] + "' from user '" + args[0] + "'.");
						}else System.out.println("Cannot remove '" + args[1] + "' from user '" + args[0] + "'.");
					}else System.out.println("Usage: removeperm <Username> <Permission>");
					break;
				case "adduser":
					if(args.length == 2) {
						if(clientManager.registerUser(args[0], Methods.sha512(args[1]))) {
							System.out.println("Registered user '" + args[0] + "'.");
						}else System.out.println("Username '" + args[0] + "' already exists!");
					}else System.out.println("Usage: adduser <Username> <Password>");
					break;
				case "removeuser":
					if(args.length == 1) {
						if(clientManager.removeUser(args[0])) {
							System.out.println("Removed user '" + args[0] + "'!");
						}else System.out.println("User '" + args[0] + "' does not exist!");
					}else System.out.println("Usage: removeuser <Username>");
					break;
				case "reload":
					executeReload();
					break;
				case "extractwebsite":
					Methods.extractWebsite();
					break;
				case "trainnow":
					HomeSystem.getAiManager().trainAll();
					break;
				default:
					throw new UnknownCommandException(command);
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void stopSystem() {
		try {
			System.out.println("Stopping server...");
			System.exit(0);
		}catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void printHelp() {
		System.out.println("Commands:");
		System.out.println("Stop -- Stops the Server");
		System.out.println("Help -- Shows this page");
		System.out.println("Version -- See the current version");
		System.out.println("Reload -- Reloads the server");
		System.out.println("ExtractWebsite -- Extracts the website files");
		System.out.println("Trainnow -- Train the AIs now");
		System.out.println("Adduser <Username> <Password> -- Add a user");
		System.out.println("Adduser <Username> -- Remove a user");
		System.out.println("Addperm <User> <Permission> -- Add a permission");
		System.out.println("Removeperm <User> <Permission> -- Remove a permission");
	}

	public static void executeReload() {
		final ClientManager clientManager = HomeSystem.getClientManager();
		final AIManager aiManager = HomeSystem.getAiManager();
		final RoomManager roomManager = HomeSystem.getRoomManager();
		final Data data = HomeSystem.getData();

		try {
			System.out.println("Reloading system...");
			HomeSystem.getPluginManager().unloadPlugins();
			HomeSystem.getScanningThread().interrupt();
			clientManager.logoutAll();
			clientManager.writeUserPerm(true);
			aiManager.stopDataSaving();
			aiManager.stopPredictions();
			aiManager.stopAutoTraining();
			aiManager.interruptAll();
			aiManager.saveData();

			data.load();
			clientManager.loadUserData();
			clientManager.loadUserPerm();
			roomManager.load();
			HomeSystem.getPluginManager().loadPlugins(new File("plugins"));
			aiManager.trainAll();
			aiManager.startDataSaving(data.getAiInterval());
			aiManager.startPredictions(data.getAiInterval());
			aiManager.startAutoTraining();

			System.out.println("Reload complete!");
			HomeSystem.startScanning();
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
}