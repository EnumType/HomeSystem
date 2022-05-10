package net.enumtype.homesystem.xmlrpc;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.server.Client;

public class XmlRpcCommand {
	
	public static void check(Client client, String command) {
		final RoomManager roomManager = Main.getRoomManager();
		if(command.equalsIgnoreCase("getrooms")) {
			client.sendMessage("rooms", roomManager.getRooms());
		}else if(command.startsWith("getdevices") && !command.startsWith("getdevicestate")
				&& !command.startsWith("getdevicetype")) {
			command = command.replaceFirst("getdevices ", "");
			String[] args = command.split(" ");
			if(args.length == 1) {
				String roomName = args[0];

				if(roomManager.existsRoom(roomName)) {
					final Room room = roomManager.getRoom(roomName);

					if(client.hasPermission(room.getPermission())) {
						client.sendMessage("roomdevices " + roomName, room.getDeviceNames());
					}else {
						client.sendMessage("noperm " + room);
					}
				}else {
					client.sendMessage("noroom " + roomName);
				}
			}else {
				client.sendMessage("failure");
			}
		}else if(command.startsWith("getdevicetype")) {
			command = command.replaceFirst("getdevicetype ", "");
			String[] args = command.split(" ");
			if(args.length == 2) {
				String roomName = args[0];
				String deviceName = args[1];

				if(roomManager.existsRoom(roomName)) {
					final Room room = roomManager.getRoom(roomName);

					if(client.hasPermission(room.getPermission())) {
						if(room.hasDevice(deviceName)) {
							client.sendMessage("device " + roomName + " " + deviceName + " " + room.getDevice(deviceName).getType());
						}else {
							client.sendMessage("nodevice " + deviceName);
						}
					}else {
						client.sendMessage("noperm " + room);
					}
				}else {
					client.sendMessage("noroom " + roomName);
				}
			}else {
				client.sendMessage("failure");
			}
		}else if(command.startsWith("getdevicestate")) {
			command = command.replaceFirst("getdevicestate ", "");
			String[] args = command.split(" ");
			if(args.length == 2) {
				String roomName = args[0];
				String deviceName = args[1];

				if(roomManager.existsRoom(roomName)) {
					final Room room = roomManager.getRoom(roomName);

					if(client.hasPermission(room.getPermission())) {
						if(room.hasDevice(deviceName)) {
							client.sendMessage("states " + deviceName, room.getDevice(deviceName).getStates());
						}else {
							client.sendMessage("nodevice " + deviceName);
						}
					}else {
						client.sendMessage("noperm " + roomName);
					}
				}else {
					client.sendMessage("noroom " + roomName);
				}
			}else {
				client.sendMessage("failure");
			}
		}else if(command.startsWith("setdevice")) {
			command = command.replaceFirst("setdevice ", "");
			String[] args = command.split(" ");

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
								device.stop(client);
							}else device.setValue(value, client);

						}else client.sendMessage("nodevice " + deviceName);
					}else client.sendMessage("noperm " + roomName);
				}else client.sendMessage("noroom " + roomName);
			}else client.sendMessage("failure");
		}
	}
	
}
