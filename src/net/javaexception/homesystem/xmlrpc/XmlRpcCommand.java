package net.javaexception.homesystem.xmlrpc;

import net.javaexception.homesystem.server.Client;

public class XmlRpcCommand {
	
	public static void check(Client client, String command) {
		if(command.equalsIgnoreCase("getrooms")) {
			client.sendMessage("rooms", Rooms.getAll());
		}else if(command.startsWith("getdevices") && !command.startsWith("getdevicestate")
				&& !command.startsWith("getdevicetype")) {
			command = command.replaceFirst("getdevices ", "");
			String[] args = command.split(" ");
			if(args.length == 1) {
				String room = args[0];
				if(Rooms.roomExists(room)) {
					if(client.hasPermission(Rooms.getRoomPerm(room))) {
						client.sendMessage("roomdevices", Rooms.getRoomDevices(room));
					}else {
						client.sendMessage("noperm " + room);
					}
				}else {
					client.sendMessage("noroom " + room);
				}


			}else {
				client.sendMessage("failure");
			}
		}else if(command.startsWith("getdevicetype")) {
			command = command.replaceFirst("getdevicetype ", "");
			String[] args = command.split(" ");
			if(args.length == 2) {
				String room = args[0];
				String device = args[1];
				if(Rooms.roomExists(room)) {
					if(client.hasPermission(Rooms.getRoomPerm(room))) {
						if(Rooms.deviceExists(room, device)) {
							client.sendMessage("device " + room + " " + device + " " + Rooms.getDeviceType(room, device));
						}else {
							client.sendMessage("nodevice " + device);
						}
					}else {
						client.sendMessage("noperm " + room);
					}
				}else {
					client.sendMessage("noroom " + room);
				}
			}else {
				client.sendMessage("failure");
			}
		}else if(command.startsWith("getdevicestate")) {
			command = command.replaceFirst("getdevicestate ", "");
			String[] args = command.split(" ");
			if(args.length == 2) {
				String room = args[0];
				String device = args[1];

				if(Rooms.roomExists(room)) {
					if(client.hasPermission(Rooms.getRoomPerm(room))) {
						if(Rooms.deviceExists(room, device)) {
							client.sendMessage("states " + device, XmlRpcServer.states(room, device, Rooms.getDeviceHmIP(room, device)));
						}else {
							client.sendMessage("nodevice " + device);
						}
					}else {
						client.sendMessage("noperm " + room);
					}
				}else {
					client.sendMessage("noroom " + room);
				}
			}else {
				client.sendMessage("failure");
			}
		}else if(command.startsWith("setdevice")) {
			command = command.replaceFirst("setdevice ", "");
			String[] args = command.split(" ");

			if(args.length == 4) {
				String room = args[0];
				String device = args[1];
				String value_key = args[2];
				Object value = args[3];

				if(args[3].equalsIgnoreCase("true") || args[3].equalsIgnoreCase("false")) {
					value = Boolean.parseBoolean(args[3]);
				}

				if(Rooms.roomExists(room)) {
					if(client.hasPermission(Rooms.getRoomPerm(room))) {
						if(Rooms.deviceExists(room, device)) {
							if(value_key.equals("STOP")) {
								XmlRpcServer.setValue(Rooms.getDeviceAddress(room, device), value_key, true, client, Rooms.getDeviceHmIP(room, device));
							}else XmlRpcServer.setValue(Rooms.getDeviceAddress(room, device), value_key, value, client, Rooms.getDeviceHmIP(room, device));
						}else client.sendMessage("nodevice " + device);
					}else client.sendMessage("noperm " + room);
				}else client.sendMessage("noroom " + room);
			}else client.sendMessage("failure");
		}
	}
	
}
