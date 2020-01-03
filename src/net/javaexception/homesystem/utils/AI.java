package net.javaexception.homesystem.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Calendar;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

import net.javaexception.homesystem.main.Main;
import net.javaexception.homesystem.xmlrpc.Rooms;
import net.javaexception.homesystem.xmlrpc.XmlRpcServer;

public class AI {
	
	private static File AIDir;
	private static File dataDir;
	private static File model;
	private static File modelPath;
	
	public static void checkAIData() throws IOException {
		AIDir = new File("AI");
		dataDir = new File(AIDir + "//data");
		model = new File(AIDir + "//Home-System.py");
		modelPath = new File(AIDir + "//models");
		
		if(!AIDir.exists()) {
			AIDir.mkdir();
		}
		
		if(!dataDir.exists()) {
			dataDir.mkdir();
		}
		
		if(!modelPath.exists()) {
			modelPath.mkdir();
		}
		
		if(!model.exists()) {
			InputStream stream = Main.class.getResourceAsStream("/Home-System.py");
			BufferedReader in = new BufferedReader(new InputStreamReader(stream, "UTF-8"));
			BufferedWriter out = new BufferedWriter(new FileWriter(model));
			
			String line;
			while((line = in.readLine()) != null) {
				out.write(line + "\r\n");
			}
			
			in.close();
			out.close();
		}
	}
	
	public static void startSavingData(int waitInMin) {
		new Thread(new Runnable() {
			@Override
			public void run() {
				while(Data.saveAIData) {
					for(String room : Rooms.getRooms()) {
						for(String device : Rooms.getRoomDevices(room)) {
							if(Rooms.getDeviceAIData(room, device)) {
								File data = new File(dataDir + "//" + room + "-" + device + ".csv");
								int brightness = 0;
								int date = Methods.getDateAsInt();
								int time = Methods.getTimeInSeconds();
								int state = 0;
								
								if(Rooms.getDeviceType(room, device).equalsIgnoreCase("ROLL")) {
									Object result = XmlRpcServer.getValue(Rooms.getDeviceAddress(room, device), "LEVEL", Rooms.getDeviceHmIP(room, device));
									
									state = Math.round(Float.parseFloat(result.toString()));
								}else if(Rooms.getDeviceType(room, device).equalsIgnoreCase("LAMP")) {
									Object result = XmlRpcServer.getValue(Rooms.getDeviceAddress(room, device), "STATE", Rooms.getDeviceHmIP(room, device));
									boolean devicestate = Boolean.parseBoolean(result.toString());
									
									state = devicestate ? 1 : 0;
								}
								
								if(!Data.aiBright.equalsIgnoreCase("AddressOfSensor") && !Data.aiBright.equalsIgnoreCase("none")) {
									Object result = XmlRpcServer.getValue(Data.aiBright, "STATE", Data.aiBrightHmIP);
									brightness = Integer.parseInt(result.toString());
								}
								
								try {
									if(data.exists()) {
										FileWriter writer = new FileWriter(data, true);
										PrintWriter out = new PrintWriter(writer);
										String line = brightness + "," + date + "," + time + "," + state;
										out.println(line);
										out.close();
									}else {
										FileWriter writer = new FileWriter(data, true);
										PrintWriter out = new PrintWriter(writer);
										String syntax = "Weather,Date,Time,State";
										String line = brightness + "," + date + "," + time + "," + state;
										out.println(syntax);
										out.println(line);
										out.close();
									}
								}catch(IOException e) {
									e.printStackTrace();
									Log.write(Methods.createPrefix() + "Error in AI(108): " + e.getMessage(), false);
								}
							}
						}
					}
					
					try {
						Thread.sleep((waitInMin * 60000));
					}catch (InterruptedException e) {
						e.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in AI(118): " + e.getMessage(), false);
					}
				}
			}
		}).start();
	}
	
	public static void startPredictions(int waitInMin) {
		new Thread(new Runnable() {
			@Override
			public void run() {
				while(Data.doAIPrediction) {
					for(String room : Rooms.getRooms()) {
						for(String device : Rooms.getRoomDevices(room)) {
							if(Rooms.getDeviceAIControll(room, device)) {
								String id = Rooms.getDeviceAddress(room, device);
								String type = Rooms.getDeviceType(room, device);
								boolean hmip = Rooms.getDeviceHmIP(room, device);
								
								if(type.equalsIgnoreCase("ROLL")) {
									int brightness = 0;
									int date = Methods.getDateAsInt();
									int time = Methods.getTimeInSeconds();
									float floatState = Float.parseFloat(XmlRpcServer.getValue(id, "LEVEL", hmip).toString());
									int state = Math.round(floatState);
									
									if(!Data.aiBright.equalsIgnoreCase("AddressOfSensor") && !Data.aiBright.equalsIgnoreCase("none")) {
										Object result = XmlRpcServer.getValue(Data.aiBright, "STATE", Data.aiBrightHmIP);
										brightness = Integer.parseInt(result.toString());
									}
									
									try {
										float prediction = (predict(room + "-" + device, brightness, date, time) / 100);
										
										if(prediction != state) {
											XmlRpcServer.setValue(id, "LEVEL", prediction, null, hmip, room);
										}
									}catch (IOException e) {
										e.printStackTrace();
										Log.write(Methods.createPrefix() + "Error in AI(157): " + e.getMessage(), false);
									}
								}else if(type.equalsIgnoreCase("LAMP")) {
									int brightness = 0;
									int date = Methods.getDateAsInt();
									int time = Methods.getTimeInSeconds();
									int state = Integer.parseInt(XmlRpcServer.getValue(id, "STATE", hmip).toString());
									
									if(!Data.aiBright.equalsIgnoreCase("AddressOfSensor") && !Data.aiBright.equalsIgnoreCase("none")) {
										Object result = XmlRpcServer.getValue(Data.aiBright, "STATE", Data.aiBrightHmIP);
										brightness = Integer.parseInt(result.toString());
									}
									
									try {
										int prediction = predict(room + "-" + device, brightness, date, time);
										
										if(prediction != state) {
											boolean targetState = (prediction > 0);
											XmlRpcServer.setValue(id, "STATE", targetState, null, hmip, room);
										}
									}catch (IOException e) {
										e.printStackTrace();
										Log.write(Methods.createPrefix() + "Error in AI(179): " + e.getMessage(), false);
									}
								}
							}
						}
					}
					
					try {
						Thread.sleep(waitInMin * 60000);
					}catch (InterruptedException e) {
						e.printStackTrace();
						Log.write(Methods.createPrefix() + "Error in AI(190): " + e.getMessage(), false);
					}
				}
			}
		}).start();
	}
	
	public static void startAutoTraining() {
		new Thread(() -> {
			Timer timer = new Timer();
			
			Calendar calendar = Calendar.getInstance();
			calendar.set(Calendar.HOUR_OF_DAY, 0);
			calendar.set(Calendar.MINUTE, 0);
			calendar.set(Calendar.SECOND, 0);
			Date time = calendar.getTime();
			
			timer.schedule(new TimerTask() {
				@Override
				public void run() {
					Log.write(Methods.createPrefix() + "Starting AI training", false);
					for(String room : Rooms.getRooms()) {
						for(String device : Rooms.getRoomDevices(room)) {
							if(Rooms.getDeviceAIData(room, device)) {
								try {
									train(room + "-" + device);
								}catch (IOException e) {
									e.printStackTrace();
									Log.write(Methods.createPrefix() + "Error in AI(218): " + e.getMessage(), false);
								}
							}
						}
					}
				}
			}, time);
		}).start();
	}
	
	public static int predict(String device, int brightness, int date, int time) throws IOException {
		int state = 0;
		if(model.exists()) {
			String cmd = "python3 " + model.getAbsolutePath() + " " +
							device + " " +
							"false " + "true " +
							"[" + brightness + "," + date + "," + time + "]";
			Process p = Runtime.getRuntime().exec(cmd);
			try {
				p.waitFor();
				
				int len;
				if ((len = p.getErrorStream().available()) > 0) {
				  byte[] buf = new byte[len];
				  p.getErrorStream().read(buf);
				  Log.write("Command error:\t\""+new String(buf)+"\"", false);
				}
				
				state = p.exitValue();
			}catch(InterruptedException e) {
				p.destroy();
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in AI(250): " + e.getMessage(), false);
			}
		}
		
		return state;
	}
	
	public static void train(String device) throws IOException {
		File data = new File("AI//data//" + device + ".csv");
		if(data.exists()) {
			String[] cmd = {"python3", "AI/Home-System.py",
							data.getName().replace(".csv", ""),
							"true",
							"false"};
			Runtime.getRuntime().exec(cmd);
		}
	}
	
}
