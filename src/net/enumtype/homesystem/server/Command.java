package net.enumtype.homesystem.server;

import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.rooms.AIManager;
import net.enumtype.homesystem.utils.*;
import net.enumtype.homesystem.rooms.Device;
import net.enumtype.homesystem.rooms.Room;
import net.enumtype.homesystem.rooms.RoomManager;
import org.eclipse.jetty.websocket.api.Session;

public class Command {
	public static void check(String command, Session session) throws UnknownCommandException {
		if(command.startsWith("console") && session == null) ConsoleCommand.check(command.replace("console", ""));
		if(session == null) return;

		final InetAddress address = session.getRemoteAddress().getAddress();
		final ClientManager clientManager = Main.getClientManager();
		command = command.toLowerCase();
		if(clientManager.isLoggedIn(address)
				|| command.startsWith("login") || command.startsWith("isonline")
				|| command.startsWith("connect")) {
			final Client client = !(command.startsWith("login") || command.startsWith("connect")) ?
					clientManager.getClient(address) : new Client(session, "", null);
			String[] args = Arrays.copyOfRange(command.split(" "), 1, command.split(" ").length);
			command = command.split(" ")[0];

			switch (command.toLowerCase()) {
				case "login":
					if(args.length == 2) {
						String user = args[0];
						String pass = args[1];

						clientManager.loginClient(session, user, pass);
					}
					break;
				case "logout":
					if(args.length == 1) {
						clientManager.logoutClient(client);
					}
					break;
				case "changeconnection":
					clientManager.getClient(session).changeConnection(true);
					break;
				case "xmlrpc":
					XmlRpcCommand.check(client, command);
					break;
				case "addperm":
					if(args.length == 2) {
						String user = args[0];
						String perm = args[1];
						clientManager.addPermission(user, perm);
					}
					break;
				case "getusername":
					client.sendMessage("user:" + client.getName());
					break;
				case "monitoring":
					MonitoringCommand.check(command.replaceFirst("monitoring ", ""), client);
					break;
				default:
					throw new UnknownCommandException(client, command);
			}

		}else {
			Main.getLog().write("Client with InetAddress: " + address + " tried to execute command: " + command,
								false, true);
			try {
				session.getRemote().sendString("notloggedin");
			}catch(IOException e) {
				Main.getLog().writeError(e);
			}
		}
	}
}

class XmlRpcCommand {
	public static void check(Client client, String command) throws UnknownCommandException {
		final RoomManager roomManager = Main.getRoomManager();
		String[] args = Arrays.copyOfRange(command.split(" "), 1, command.split(" ").length);
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
						}else client.sendMessage("noperm " + room);
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
	public static void check(String command, Client client) {
		if(client.hasPermission("system.monitoring")) {
			final Monitoring monitoring = Main.getMonitoring();

			if (command.startsWith("isXmlRpcReachable")) {
				command = command.replace("isXmlRpcReachable", "");
				String[] args = command.split(" ");
				if (args.length == 1) {
					int timeout = Integer.parseInt(args[0]);
					client.sendMessage("xmlrpc " + monitoring.isXmlRpcReachable(timeout));
				}
			}
		}else client.sendMessage("noperm");
	}
}

class ConsoleCommand {
	public static void check(String command) {
		final ClientManager clientManager = Main.getClientManager();
		final Log log = Main.getLog();
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
					log.write("Current version of Home-System: " + Main.getData().getVersion(), false, true);
					break;
				case "addperm":
					if(args.length != 2) {
						System.out.println("Usage: >addperm <Username> <Permission>");
						break;
					}
					clientManager.addPermission(args[0], args[1]);
					log.write("Added permission '" + args[1] + "' to user '" + args[0] + "'.", false, true);
					break;
				case "removeperm":
					if(args.length == 2) {
						if(clientManager.removePermission(args[0], args[1])) {
							log.write("Removed '" + args[1] + "' from user '" + args[0] + "'.", false, true);
						}else log.write("Cannot remove '" + args[1] + "' from user '" + args[0] + "'.", false, true);
					}else System.out.println("Usage: >removeperm <Username> <Permission>");
					break;
				case "adduser":
					if(args.length == 2) {
						clientManager.registerUser(args[0], Methods.sha512(args[1]));
						log.write("Registered user '" + args[0] + "'.", false, true);
					}else System.out.println("Usage: >adduser <Username> <Password>");
				case "reload":
					executeReload();
					break;
				case "extract website":
					Methods.extractWebsite();
					break;
				case "train now":
					Main.getAiManager().trainAll();
					break;
			}
		}catch(Exception e) {
			log.writeError(e);
		}
	}

	public static void stopSystem() {
		try {
			Main.getLog().write("Stopping server...", true, true);
			Main.getClientManager().writeUserPerm(true);
			Main.getScanningThread().interrupt();
			Main.getWsServer().stop();
			System.exit(0);
		}catch(Exception e) {
			Main.getLog().writeError(e);
		}
	}

	public static void printHelp() {
		final Log log = Main.getLog();

		log.write("Commands:", true, false);
		log.write("Stop -- Stops the Server", true, false);
		log.write("Help -- Shows this page", true, false);
		log.write("Version <version> -- Change the version", true, false);
		log.write("Addperm <User> <Permission> -- Add a permission", true, false);
		log.write("Extract Website -- Extracts the Webinterface", true, false);
		log.write("", true, false);
	}

	public static void executeReload() {
		final ClientManager clientManager = Main.getClientManager();
		final AIManager aiManager = Main.getAiManager();
		final RoomManager roomManager = Main.getRoomManager();
		final Data data = Main.getData();
		final Log log = Main.getLog();

		try {
			log.write("Reloading system...", true, true);
			Main.getScanningThread().interrupt();
			aiManager.stopDataSaving();
			aiManager.stopPredictions();
			aiManager.stopAutoTraining();
			clientManager.writeUserPerm(true);

			clientManager.loadUserData();
			clientManager.loadUserPerm();
			data.load();
			roomManager.load();
			aiManager.startDataSaving(data.getAiInterval());
			aiManager.startPredictions(data.getAiInterval());
			aiManager.startAutoTraining();

			log.write("Reload complete!", false, true);
			Main.startScanning();
		}catch(Exception e) {
			log.writeError(e);
		}
	}
}