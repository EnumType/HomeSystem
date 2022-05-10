package net.enumtype.homesystem.monitoring;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import net.enumtype.homesystem.utils.Data;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;

public class Monitoring {
	
	private final Map<String, String> errors;

	public Monitoring() {
		errors = new HashMap<>();
	}

	public Map<String, String> getErrors() {return errors;}
	
	public ArrayList<String> getLog() {
		ArrayList<String> list = new ArrayList<>();
		File latest = new File("logs/latest.log");

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
		
		return list;
	}
	
	public boolean isXmlRpcReachable(int timeout) {
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
