package net.enumtype.homesystem.server;

import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.List;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.utils.Monitoring;
import net.enumtype.homesystem.utils.UnknownCommandException;
import net.enumtype.homesystem.rooms.Device;
import net.enumtype.homesystem.rooms.Room;
import net.enumtype.homesystem.rooms.RoomManager;
import org.eclipse.jetty.websocket.api.Session;

public class Command {
	
	public static void check(String command, Session session) throws UnknownCommandException {
		final InetAddress address = session.getRemoteAddress().getAddress();
		final ClientManager clientManager = Main.getClientManager();
		command = command.toLowerCase();
		if(clientManager.isLoggedIn(address)
				|| command.startsWith("login") || command.startsWith("isonline")
				|| command.startsWith("connect")) {
			final Client client = !(command.startsWith("login") || command.startsWith("connect")) ?
					clientManager.getClient(address) : new Client(session, "", null);
			String[] args;

			switch (command.split(" ")[0]) {
				case "login":
					command = command.replace("login ", "");
					args = command.split(" ");

					if(args.length == 2) {
						String user = args[0];
						String pass = args[1];

						clientManager.loginClient(session, user, pass);
					}
					break;
				case "logout":
					command = command.replace("logout ", "");

					args = command.split(" ");

					if(args.length == 1) {
						clientManager.logoutClient(client);
					}
					break;
				case "changeconnection":
					clientManager.getClient(session).changeConnection(true);
					break;
				case "xmlrpc":
					command = command.replaceFirst("xmlrpc ", "");
					XmlRpcCommand.check(client, command);
					break;
				case "addperm":
					command = command.replaceFirst("addperm ", "");
					args = command.split(" ");

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
				if(Main.getData().printStackTraces()) e.printStackTrace();
				Main.getLog().writeError(e);
			}
		}
	}
	
}

class XmlRpcCommand {
	public static void check(Client client, String command) throws UnknownCommandException {
		final RoomManager roomManager = Main.getRoomManager();
		String[] args;

		switch (command.split(" ")[0].toLowerCase()) {
			case "getrooms":
				final List<String> rooms = new ArrayList<>();

				roomManager.getRooms().forEach(room -> rooms.add(room.getName()));

				client.sendMessage("rooms", rooms);
				break;
			case "getdevices":
				args = command.replaceFirst("getdevices ", "").split(" ");
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
				args = command.replaceFirst("getdevicetype ", "").split(" ");

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
				args = command.replaceFirst("getdevicestate ", "").split(" ");

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
				args = command.replaceFirst("setdevice ", "").split(" ");

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