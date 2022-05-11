package net.enumtype.homesystem.xmlrpc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Map;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.DumperOptions.FlowStyle;

public class Rooms {
	
	public static Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> rooms;
	
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
