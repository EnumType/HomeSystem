package net.javaexception.homesystem.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.DumperOptions.FlowStyle;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.server.Client;

public class UserData {
	
	public static void loadUserData() {
		if(!Client.users.isEmpty()) {
			Client.users.clear();
		}
		
		File file = new File("User-Data");
		File dataFile = new File(file + "//data.txt");
		if(!file.exists()) {
			file.mkdirs();
		}
		if(!dataFile.exists()) {
			registerUser("a", "a");
			removeUser("a", "a");
		}
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(file + "//data.txt"));
			boolean end = false;
			while(!end) {
				String user = reader.readLine();
				String pass = reader.readLine();
				
				if(user == null && pass == null) {
					end = true;
				}else if(user.startsWith("Name: ") && pass.startsWith("Password: ")){
					Client.users.put(user.replaceAll("Name: ", ""), pass.replaceAll("Password: ", ""));
				}
			}
			
			reader.close();
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in UserData(57): " + e.getMessage(), false);
		}
	}
	
	public static void registerUser(String username, String password) {
		try {
			PrintWriter out = new PrintWriter(new FileWriter("User-Data//data.txt", true), true);
			out.write("Name: " + username + "\r\n");
			out.write("Password: " + password + "\r\n");
			out.close();
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in UserData(69): " + e.getMessage(), false);
		}
	}
	
	public static void removeUser(String username, String password) {
		try {
			File data = new File("User-Data//data.txt");
			File newData = new File("User-Data//data2.txt");
			BufferedReader reader = new BufferedReader(new FileReader(data));
			BufferedWriter writer = new BufferedWriter(new FileWriter(newData));
			boolean found = false;
			String line;
			
			while((line = reader.readLine()) != null) {
				if(!found) {
					if(!line.equals("Name: " + username) && !line.equals("Password: " + password) && !line.equals("\r\n")) {
						writer.write(line + "\r\n");
					}else {
						found = true;
						reader.readLine();
					}
				}else {
					writer.write(line + "\r\n");
				}
			}
			
			reader.close();
			writer.close();
			data.delete();
			newData.renameTo(data);
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in UserData(101): " + e.getMessage(), false);
		}
	}
	
	@SuppressWarnings("unchecked")
	public static void loadUserPerm() {
		try {
			File file = new File("Permissions.yml");
			if(file.exists()) {
				if(!Client.permissions.isEmpty()) {
					Client.permissions.clear();
				}
				
				Log.write(Methods.createPrefix() + "Loading Permissions.yml...", true);
				Yaml yaml = new Yaml();
				FileInputStream io = new FileInputStream(new File("Permissions.yml"));
				
				Map<Object, List<Object>> list = (Map<Object, List<Object>>)yaml.load(io);
				
				list.keySet().forEach(user -> {
					list.get(user).forEach(permission -> {
						Client.addPermission(user.toString(), permission.toString());
					});
				});
			}else {
				try {
					Log.write(Methods.createPrefix() + "Creating Permissions.yml...", true);					
					InputStream resource = Main.class.getResourceAsStream("/Permissions.yml");
					Yaml in = new Yaml();
					Map<String, List<String>> map = (Map<String, List<String>>) in.load(resource);
					
					DumperOptions options = new DumperOptions();
					options.setDefaultFlowStyle(FlowStyle.BLOCK);
					options.setPrettyFlow(true);
					
					Yaml out = new Yaml(options);
					FileWriter writer = new FileWriter(file);
					out.dump(map, writer);
					loadUserPerm();
				}catch(IOException e) {
					e.printStackTrace();
					Log.write(Methods.createPrefix() + "Error in UserData(142): " + e.getMessage(), false);
				}
			}
		}catch(FileNotFoundException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in UserData(147): " + e.getMessage(), false);
		}
	}
	
	public static void writeUserPerm() {
		Log.write(Methods.createPrefix() + "Saving permissions...", true);
		try {
			DumperOptions options = new DumperOptions();
			options.setDefaultFlowStyle(FlowStyle.BLOCK);
			options.setPrettyFlow(true);
			
			Yaml yaml = new Yaml(options);
			FileWriter writer = new FileWriter(new File("Permissions.yml"));
			yaml.dump(Client.permissions, writer);
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(Methods.createPrefix() + "Error in UserData(163): " + e.getMessage(), false);
		}
	}
	
}
