package net.enumtype.homesystem.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;

import net.enumtype.homesystem.HomeSystem;

public class Monitoring {
	
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
		}
		
		return list;
	}
	
	public boolean isXmlRpcReachable(int timeout) {
		try {
			return InetAddress.getByName(HomeSystem.getData().getXmlRpcAddress()).isReachable(timeout);
		}catch(IOException e) {
			e.printStackTrace();
		}
		return false;
	}
	
}
