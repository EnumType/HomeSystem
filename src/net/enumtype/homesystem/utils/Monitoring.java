package net.enumtype.homesystem.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;

import net.enumtype.homesystem.Main;

public class Monitoring {

	private final Log log;

	public Monitoring() {
		log = Main.getLog();
	}
	
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
			log.writeError(e);
		}
		
		return list;
	}
	
	public boolean isXmlRpcReachable(int timeout) {
		try {
			return InetAddress.getByName(Main.getData().getXmlRpcAddress()).isReachable(timeout);
		}catch(IOException e) {
			log.writeError(e);
		}
		return false;
	}
	
}
