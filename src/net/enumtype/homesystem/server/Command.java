package net.enumtype.homesystem.server;

import java.net.InetAddress;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.monitoring.MonitoringCommand;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import net.enumtype.homesystem.websocket.WebSocket;
import net.enumtype.homesystem.xmlrpc.XmlRpcCommand;
import org.eclipse.jetty.websocket.api.Session;

public class Command {
	
	public static void check(String command, Session session) {
		final InetAddress address = session.getRemoteAddress().getAddress();
		command = command != null ? command.toLowerCase() : "";
		if(Main.getClientManager().isLoggedIn(address)
				|| command.startsWith("login") || command.startsWith("isonline")
				|| command.startsWith("connect")) {
			final ClientManager clientManager = Main.getClientManager();
			final Client client = !(command.startsWith("login") || command.startsWith("connect")) ?
					clientManager.getClient(address) : new Client(session, "", null);
			String[] args;

			switch (command) {
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
					Log.write("Error in Command switch", false);
			}

		}else {
			Log.write(Methods.createPrefix() + "Client with InetAddress: " + address + " tried to execute command: " + command, false);
			WebSocket.sendCommand("notloggedin", session);
		}
	}
	
}
