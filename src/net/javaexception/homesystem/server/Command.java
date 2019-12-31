package net.javaexception.homesystem.server;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Base64;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.utils.Crypto;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;
import net.javaexception.homesystem.xmlrpc.XmlRpcCommand;

public class Command {
	
	public static void checkCommand(String command, InetAddress address) {
		if(Client.isLoggedIn(address)
				|| command.startsWith("login") || command.startsWith("changedAddress")
				|| command.startsWith("isonline") || command.startsWith("connect") || command.startsWith("clientkey")) {
			if(command != null) {
				if(command.equalsIgnoreCase("stop")) {
					Commands.executeStopCommand();
				}else if(command.equalsIgnoreCase("help")) {
					Commands.executeHelpCommand();
				}else if(command.startsWith("login")) {
					command = command.replace("login ", "");
					
					String user = "";
					String pass = "";
					
					String[] args = command.split(" ");
					
					if(args.length == 2) {
						user = args[0];
						pass = args[1];
						
						Client.loginClient(user, pass, address);
					}
				}else if(command.startsWith("logout")) {
					command = command.replace("logout ", "");
					
					String[] args = command.split(" ");
					
					if(args.length == 1) {
						Client.logoutClient(args[0], address);
					}
				}else if(command.startsWith("xmlrpc")) {
					command = command.replaceFirst("xmlrpc ", "");
					XmlRpcCommand.checkCommand(address, command, Main.crypto);
				}else if(command.startsWith("whoisonline")) {
					try {
						Server.sendCommand(address, "onlineusers " + Client.getOnlineUsers());
					} catch (UnknownHostException e) {
						e.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in Command(55): " + e.getMessage(), false);
					} catch (IOException e) {
						e.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in Command(58): " + e.getMessage(), false);
					}
				}else if(command.startsWith("addperm")) {
					command = command.replaceFirst("addperm ", "");
					String[] args = command.split(" ");
					
					if(args.length == 2) {
						String user = args[0];
						String perm = args[1];
						Client.addPermission(user, perm);
					}
				}else if(command.startsWith("clientkey")) {
					try {
						Crypto crypto = Main.crypto;
						byte[] pubkey = crypto.getPublicKey().getEncoded();
						
						command = command.replaceFirst("clientkey: ", "");
						String[] args = command.split(" ");
						String pk = Base64.getEncoder().encodeToString(pubkey);
						if(Client.userKeyExists(address)) {
							Client.removeUserKey(address);
						}
						
						if(args.length == 1) {
							byte[] clientkey = Base64.getDecoder().decode(args[0]);
							
							Server.sendCommand(address, "serverkey: " + pk);
							Client.addUserKey(address, crypto.readPublicKey(clientkey));
						}
					} catch (IOException e) {
						e.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in Command(89): " + e.getMessage(), false);
					}
				}else if(command.startsWith("getusername")) {
					try {
						Server.sendCommand(address, "user:" + Client.getUsername(address));
					} catch (IOException e) {
						e.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in Command(96): " + e.getMessage(), true);
						Log.write("", false);
					}
				}
			}
			
		}else {
			Log.write(Methods.createPrefix() + "Client with InetAddress: " + address
				+ " tried to execute command: " + command, true);
			Log.write("", false);
			try {
				Server.sendCommand(address, "notloggedin");
			} catch (IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Command(110): " + e.getMessage(), true);
				Log.write("", false);
			}
		}
	}
	
}
