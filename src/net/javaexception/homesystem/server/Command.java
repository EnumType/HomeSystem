package net.javaexception.homesystem.server;

import java.net.InetAddress;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.monitoring.MonitoringCommand;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;
import net.javaexception.homesystem.websocket.WebSocket;
import net.javaexception.homesystem.xmlrpc.XmlRpcCommand;

public class Command {
	
	public static void checkCommand(String command, InetAddress address) {
		if(Main.getClientManager().isLoggedIn(address)
				|| command.startsWith("login") || command.startsWith("isonline")
				|| command.startsWith("connect")) {
			if(command != null) {
				final ClientManager clientManager = Main.getClientManager();
				final Client client = !(command.startsWith("login") || command.startsWith("connect")) ?
						clientManager.getClient(address) : new Client(address, "", null);

				if(command.equalsIgnoreCase("stop")) {
					Commands.executeStopCommand();
				}else if(command.equalsIgnoreCase("help")) {
					Commands.executeHelpCommand();
				}else if(command.startsWith("login")) {
					command = command.replace("login ", "");
					String[] args = command.split(" ");
					
					if(args.length == 2) {
						String user = args[0];
						String pass = args[1];

						clientManager.loginClient(address, user, pass);
					}
				}else if(command.startsWith("logout")) {
					command = command.replace("logout ", "");
					
					String[] args = command.split(" ");
					
					if(args.length == 1) {
						clientManager.logoutClient(client);
					}
				}else if(command.startsWith("xmlrpc")) {
					command = command.replaceFirst("xmlrpc ", "");
					XmlRpcCommand.checkCommand(client, command);
				}else if(command.startsWith("whoisonline")) {
					client.sendMessage("onlineusers " + clientManager.getOnlineUsers());
				}else if(command.startsWith("addperm")) { //TODO: Security check
					command = command.replaceFirst("addperm ", "");
					String[] args = command.split(" ");
					
					if(args.length == 2) {
						String user = args[0];
						String perm = args[1];
						clientManager.addPermission(user, perm);
					}
				}else if(command.startsWith("getusername")) {
					client.sendMessage("user:" + client.getName());
				}else if(command.startsWith("monitoring")) {
					MonitoringCommand.checkCommand(command.replaceFirst("monitoring ", ""), client);
				}
			}
			
		}else {
			Log.write(Methods.createPrefix() + "Client with InetAddress: " + address + " tried to execute command: " + command, false);
			WebSocket.sendCommand("notloggedin", address);
		}
	}
	
}
