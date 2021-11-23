package net.javaexception.homesystem.xmlrpc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Map;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.DumperOptions.FlowStyle;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.utils.Log;
import net.javaexception.homesystem.utils.Methods;

public class Rooms {
	
	public static Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> rooms;
	
	@SuppressWarnings("unchecked")
	public static void loadData() {
		try {
			if(rooms != null) {
				rooms.clear();
			}
			
			File file = new File("Rooms.yml");
			if(file.exists()) {
				Log.write(Methods.createPrefix() + "Loading Rooms.yml...", true);
				FileInputStream in = new FileInputStream(file);
				Yaml yaml = new Yaml();
				rooms = (Map<Object, Map<Object, Map<Object, Map<Object, Object>>>>) yaml.load(in);
			}else {
				Log.write(Methods.createPrefix() + "Creating Rooms.yml...", true);
				InputStream resource = Main.class.getResourceAsStream("/Rooms.yml");
				Yaml in = new Yaml();
				Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> map = (Map<Object, Map<Object, Map<Object, Map<Object, Object>>>>) in.load(resource);
				
				DumperOptions options = new DumperOptions();
				options.setDefaultFlowStyle(FlowStyle.BLOCK);
				options.setPrettyFlow(true);
				
				Yaml out = new Yaml(options);
				FileWriter writer = new FileWriter(file);
				out.dump(map, writer);
				loadData();
			}
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in Rooms(53): " + e.getMessage(), false);
		}
	}
	
	public static boolean roomExists(String room) {
		return rooms.get(room) != null;
	}
	
	public static boolean deviceExists(String room, String device) {
		return rooms.get(room).get("Devices").get(device) != null;
	}
	
	public static String getRoomPerm(String room) {
		Object o = rooms.get(room).get("Permission");
		return o.toString();
	}
	
	public static String getDeviceType(String room, String device) {
		return rooms.get(room).get("Devices").get(device).get("Type").toString();
	}
	
	public static String getDeviceAddress(String room, String device) {
		return rooms.get(room).get("Devices").get(device).get("Address").toString();
	}
	
	public static boolean getDeviceAIData(String room, String device) {
		return Boolean.parseBoolean(rooms.get(room).get("Devices").get(device).get("AIData").toString());
	}
	
	public static boolean getDeviceAIControll(String room, String device) {
		return Boolean.parseBoolean(rooms.get(room).get("Devices").get(device).get("AIControl").toString());
	}
	
	public static boolean getDeviceHmIP(String room, String device) {
		return Boolean.parseBoolean(rooms.get(room).get("Devices").get(device).get("HmIP").toString());
	}
	
	public static ArrayList<String> getAll() {
		ArrayList<String> list = new ArrayList<>();
		for(Object o : rooms.keySet()) {
			list.add(o.toString());
		}
		
		return list;
	}
	
	public static ArrayList<String> getRoomDevices(String room) {
		ArrayList<String> list = new ArrayList<>();
		rooms.get(room).get("Devices").keySet().forEach(o -> list.add(o.toString()));
		
		return list;
	}
	
}
