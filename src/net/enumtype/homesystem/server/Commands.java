package net.enumtype.homesystem.server;

import java.io.IOException;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import net.enumtype.homesystem.utils.Data;

public class Commands {
	
	public static void executeStopCommand() {
		try {
			Main.getLog().write("Stopping server...", true);
			Main.getClientManager().writeUserPerm();
			System.exit(0);
		}catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void updateVersion(String version) {
		Main.getData().setVersion(version);
		Main.getLog().write(Methods.createPrefix() + "Version changed to " + version, false);
	}
	
	public static void executeHelpCommand() {
		final Log log = Main.getLog();

		log.write("Commands:", true);
		log.write("Stop -- Stops the Server", true);
		log.write("Help -- Shows this page", true);
		log.write("Version <version> -- Change the version", true);
		log.write("Addperm <User> <Permission> -- Add a permission", true);
		log.write("Extract Website -- Extracts the Webinterface", true);
		log.write("", true);
	}
	
}
