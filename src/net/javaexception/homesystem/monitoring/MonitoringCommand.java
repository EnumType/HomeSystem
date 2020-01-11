package net.javaexception.homesystem.monitoring;

import java.io.IOException;
import java.net.InetAddress;

import net.javaexception.homesystem.server.Server;

public class MonitoringCommand {
	
	public static void checkCommand(String command, InetAddress address) {
		if(command.startsWith("isXmlRpcReachable")) {
			try {
				command = command.replace("isXmlRpcReachable", "");
				String[] args = command.split(" ");
				if(args.length == 1) {
					int timeout = Integer.parseInt(args[0]);
					Server.sendCommand(address, "xmlrpc " + Monitoring.isXmlRpcReachable(timeout));
				}
			}catch(IOException e) {
				e.printStackTrace();
			}
		}
	}
	
}
