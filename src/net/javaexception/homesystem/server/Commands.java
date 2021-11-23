package net.javaexception.homesystem.server;

import java.io.IOException;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.utils.Data;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

public class Commands {
	
	public static void executeStopCommand() {
		try {
			Log.write("Stopping server...", true);
			Main.getClientManager().writeUserPerm();
			System.exit(0);
		}catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void updateVersion(String version) {
		Data.version = version;
		Log.write(Methods.createPrefix() + "Version changed to " + Data.version, false);
	}
	
	public static void executeHelpCommand() {
		Log.write("Commands:", true);
		Log.write("Stop -- Stops the Server", true);
		Log.write("Help -- Shows this page", true);
		Log.write("Version <version> -- Change the version", true);
		Log.write("Addperm <User> <Permission> -- Add a permission", true);
		Log.write("Extract Website -- Extracts the Webinterface", true);
		Log.write("", true);
	}
	
}
