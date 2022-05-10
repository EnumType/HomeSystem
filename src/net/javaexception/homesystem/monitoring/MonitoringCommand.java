package net.javaexception.homesystem.monitoring;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.server.Client;

public class MonitoringCommand {
	
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
