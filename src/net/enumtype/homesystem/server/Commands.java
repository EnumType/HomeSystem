package net.enumtype.homesystem.server;

import java.io.IOException;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.utils.Log;

public class Commands {
	
	public static void executeStopCommand() {
		try {
			Main.getLog().write("Stopping server...", true, true);
			Main.getClientManager().writeUserPerm();
			System.exit(0);
		}catch(IOException e) {
			if(Main.getData().printStackTraces()) e.printStackTrace();
			Main.getLog().writeError(e);
		}
	}
	
	public static void updateVersion(String version) {
		Main.getData().setVersion(version);
		Main.getLog().write("Version changed to " + version, false, true);
	}
	
	public static void executeHelpCommand() {
		final Log log = Main.getLog();

		log.write("Commands:", true, false);
		log.write("Stop -- Stops the Server", true, false);
		log.write("Help -- Shows this page", true, false);
		log.write("Version <version> -- Change the version", true, false);
		log.write("Addperm <User> <Permission> -- Add a permission", true, false);
		log.write("Extract Website -- Extracts the Webinterface", true, false);
		log.write("", true, false);
	}
	
}
