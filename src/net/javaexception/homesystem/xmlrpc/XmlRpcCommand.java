package net.javaexception.homesystem.xmlrpc;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;

import net.javaexception.homesystem.server.Client;
import net.javaexception.homesystem.server.Server;
import net.javaexception.homesystem.utils.Crypto;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

public class XmlRpcCommand {
	
	public static void checkCommand(InetAddress address, String command, Crypto crypto) {
		try {
			String user = Client.getUsername(address);
			
			if(command.equalsIgnoreCase("getrooms")) {
				Server.sendPackageCommand(address, Rooms.getRooms(), "rooms", crypto);
			}else if(command.startsWith("getdevices") && !command.startsWith("getdevicestate")
						&& !command.startsWith("getdevicetype")) {
				command = command.replaceFirst("getdevices ", "");
				String[] args = command.split(" ");
				if(args.length == 1) {
					String room = args[0];
					if(Rooms.roomExists(room)) {
						if(Client.hasPermission(user, Rooms.getRoomPerm(room))) {
							Server.sendPackageCommand(address, Rooms.getRoomDevices(room), "roomdevices:" + room, crypto);
						}else {
							Server.sendCommand(address, "noperm " + room);
						}
					}else {
						Server.sendCommand(address, "noroom " + room);
					}
				}else {
					Server.sendCommand(address, "failure");
				}
			}else if(command.startsWith("getdevicetype")) {
				command = command.replaceFirst("getdevicetype ", "");
				String[] args = command.split(" ");
				if(args.length == 2) {
					String room = args[0];
					String device = args[1];
					if(Rooms.roomExists(room)) {
						if(Client.hasPermission(user, Rooms.getRoomPerm(room))) {
							if(Rooms.deviceExists(room, device)) {
								Server.sendCommand(address, "device " + room + " " + device + " " + Rooms.getDeviceType(room, device));
							}else {
								Server.sendCommand(address, "nodevice " + device);
							}
						}else {
							Server.sendCommand(address, "noperm " + room);
						}
					}else {
						Server.sendCommand(address, "noroom " + room);
					}
				}else {
					Server.sendCommand(address, "failure");
				}
			}else if(command.startsWith("getdevicestate")) {
				command = command.replaceFirst("getdevicestate ", "");
				String[] args = command.split(" ");
				if(args.length == 2) {
					String room = args[0];
					String device = args[1];
					
					if(Rooms.roomExists(room)) {
						if(Client.hasPermission(user, Rooms.getRoomPerm(room))) {
							if(Rooms.deviceExists(room, device)) {
								Server.sendPackageCommand(address, XmlRpcServer.states(room, device, Rooms.getDeviceHmIP(room, device)), "states " + device, crypto);
							}else {
								Server.sendCommand(address, "nodevice " + device);
							}
						}else {
							Server.sendCommand(address, "noperm " + room);
						}
					}else {
						Server.sendCommand(address, "noroom " + room);
					}
				}else {
					Server.sendCommand(address, "failure");
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
						if(Client.hasPermission(user, Rooms.getRoomPerm(room))) {
							if(Rooms.deviceExists(room, device)) {
								if(value_key.equals("STOP")) {
									XmlRpcServer.setValue(Rooms.getDeviceAddress(room, device), value_key, true, address, Rooms.getDeviceHmIP(room, device), room);
								}else {
									XmlRpcServer.setValue(Rooms.getDeviceAddress(room, device), value_key, value, address, Rooms.getDeviceHmIP(room, device), room);
								}
							}else {
								Server.sendCommand(address, "nodevice " + device);
							}
						}else Server.sendCommand(address, "noperm " + room);
					}else {
						Server.sendCommand(address, "noroom " + room);
					}
				}else {
					Server.sendCommand(address, "failure");
				}
			}
		} catch (UnknownHostException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in XmlRpcCommand(119): " + e.getMessage(), false);
		} catch (IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in XmlRpcCommand(122): " + e.getMessage(), false);
		}
	}
	
}
