package net.javaexception.homesystem.monitoring;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.utils.Data;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

public class Monitoring {
	
	private static Map<String, String> errors;
	
	public static void init() {
		errors = new HashMap<>();
	}
	
	public static Map<String, String> getErrors() {
		return errors;
	}
	
	public static ArrayList<String> getLog() {
		ArrayList<String> list = new ArrayList<>();
		File latest = new File("logs/latest.log");
		
		if(latest != null) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(latest));
				
				String line;
				while((line = reader.readLine()) != null) {
					list.add(line);
				}
				
				reader.close();
			}catch(IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in Monitoring(42): " + e.getMessage(), false);
			}
		}
		
		return list;
	}
	
	public static void executeCommand(String command) {
		Main.checkConsoleCommand(command);
	}
	
	public static boolean isXmlRpcReachable(int timeout) {
		try {
			String host = Data.xmlrpcAddress;
			return InetAddress.getByName(host).isReachable(timeout);
		} catch (IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Monitoring(): " + e.getMessage(), false);
		}
		return false;
	}
	
}
