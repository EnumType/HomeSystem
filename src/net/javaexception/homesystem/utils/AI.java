package net.javaexception.homesystem.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

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
		}else {
			model.delete();
			checkAIData();
		}
	}
	
	public static void startSavingData(int waitInMin) {
		Timer timer = new Timer();
		
		timer.scheduleAtFixedRate(new TimerTask() {
			@Override
			public void run() {
				if(Data.saveAIData) {
					for(String room : Rooms.getRooms()) {
						for(String device : Rooms.getRoomDevices(room)) {
							if(Rooms.getDeviceAIData(room, device)) {
								File data = new File(dataDir + "//" + room + "-" + device + ".csv");
								int brightness = 0;
								long time = Methods.getUnixTime();
								int state = 0;
								
								if(Rooms.getDeviceType(room, device).equalsIgnoreCase("ROLL")) {
									Object result = XmlRpcServer.getValue(Rooms.getDeviceAddress(room, device), "LEVEL", Rooms.getDeviceHmIP(room, device));
									
									if(!result.toString().isEmpty()) {
										state = Math.round(Float.parseFloat(result.toString()));
									}
								}else if(Rooms.getDeviceType(room, device).equalsIgnoreCase("LAMP")) {
									Object result = XmlRpcServer.getValue(Rooms.getDeviceAddress(room, device), "STATE", Rooms.getDeviceHmIP(room, device));
									
									if(!result.toString().isEmpty()) {
										boolean devicestate = Boolean.parseBoolean(result.toString());
										state = devicestate ? 1 : 0;
									}
								}
								
								if(!Data.aiBright.equalsIgnoreCase("AddressOfSensor") && !Data.aiBright.equalsIgnoreCase("none")) {
									Object result = XmlRpcServer.getValue(Data.aiBright, "STATE", Data.aiBrightHmIP);
									brightness = Integer.parseInt(result.toString());
								}
								
								try {
									if(data.exists()) {
										FileWriter writer = new FileWriter(data, true);
										PrintWriter out = new PrintWriter(writer);
										String line = brightness + "," + time + "," + state;
										out.println(line);
										writer.flush();
										writer.close();
										out.close();
									}else {
										FileWriter writer = new FileWriter(data, true);
										PrintWriter out = new PrintWriter(writer);
										String syntax = "Weather,Time,State";
										String line = brightness + "," + time + "," + state;
										out.println(syntax);
										out.println(line);
										writer.flush();
										writer.close();
										out.close();
									}
								}catch(IOException e) {
									e.printStackTrace();
									Log.write(Methods.createPrefix() + "Error in AI(123): " + e.getMessage(), false);
								}
							}
						}
					}
				}
			}
		}, 0, waitInMin * 60000);
	}
	
	public static void startPredictions(int waitInMin) {
		Timer timer = new Timer();
		
		timer.scheduleAtFixedRate(new TimerTask() {
			@Override
			public void run() {
				if(Data.doAIPrediction) {
					for(String room : Rooms.getRooms()) {
						for(String device : Rooms.getRoomDevices(room)) {
							if(Rooms.getDeviceAIControll(room, device)) {
								String id = Rooms.getDeviceAddress(room, device);
								String type = Rooms.getDeviceType(room, device);
								boolean hmip = Rooms.getDeviceHmIP(room, device);
								
								if(type.equalsIgnoreCase("ROLL")) {
									int brightness = 0;
									long time = Methods.getUnixTime();
									float floatState = 1;
									Object rollstate = XmlRpcServer.getValue(id, "LEVEL", hmip);
									
									if(!rollstate.toString().isEmpty()) {
										floatState = Float.parseFloat(rollstate.toString());
									}
									
									int state = Math.round(floatState);
									
									if(!Data.aiBright.equalsIgnoreCase("AddressOfSensor") && !Data.aiBright.equalsIgnoreCase("none")) {
										Object result = XmlRpcServer.getValue(Data.aiBright, "STATE", Data.aiBrightHmIP);
										
										if(!result.toString().isEmpty()) {
											brightness = Integer.parseInt(result.toString());
										}
									}
									
									try {
										float prediction = (predict(room + "-" + device, brightness, time) / 100);
										
										if(prediction != state) {
											XmlRpcServer.setValue(id, "LEVEL", prediction, null, hmip, room);
										}
									}catch (IOException e) {
										e.printStackTrace();
										Log.write(Methods.createPrefix() + "Error in AI(175): " + e.getMessage(), false);
									}
								}else if(type.equalsIgnoreCase("LAMP")) {
									int brightness = 0;
									long time = Methods.getUnixTime();
									int state = 0;
									
									Object lampstate = XmlRpcServer.getValue(id, "STATE", hmip);
									
									if(!lampstate.toString().isEmpty()) {
										state = Integer.parseInt(lampstate.toString());
									}
									
									if(!Data.aiBright.equalsIgnoreCase("AddressOfSensor") && !Data.aiBright.equalsIgnoreCase("none")) {
										Object result = XmlRpcServer.getValue(Data.aiBright, "STATE", Data.aiBrightHmIP);
										
										if(!result.toString().isEmpty()) {
											brightness = Integer.parseInt(result.toString());
										}
									}
									
									try {
										int prediction = predict(room + "-" + device, brightness, time);
										
										if(prediction != state) {
											boolean targetState = (prediction > 0);
											XmlRpcServer.setValue(id, "STATE", targetState, null, hmip, room);
										}
									}catch (IOException e) {
										e.printStackTrace();
										Log.write(Methods.createPrefix() + "Error in AI(205): " + e.getMessage(), false);
									}
								}
							}
						}
					}
				}
			}
		}, 0, waitInMin * 60000);
	}
	
	public static void startAutoTraining() {
		new Thread(() -> {
			ZonedDateTime now = ZonedDateTime.now();
			ZonedDateTime nextRun = now.withHour(0).withMinute(0).withSecond(0);
			
			if(now.compareTo(nextRun) > 0) {
				nextRun = nextRun.plusDays(1);
			}
			
			Duration duration = Duration.between(now, nextRun);
			long initialDelay = duration.getSeconds();
			
			ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
			scheduler.scheduleAtFixedRate(() -> {
				Log.write(Methods.createPrefix() + "Starting AI training", false);
				for(String room : Rooms.getRooms()) {
					for(String device : Rooms.getRoomDevices(room)) {
						if(Rooms.getDeviceAIData(room, device)) {
							train(room + "-" + device);
						}
					}
				}
			}, initialDelay, TimeUnit.DAYS.toSeconds(1), TimeUnit.SECONDS);
		}).start();
	}
	
	public static int predict(String device, int brightness, long time) throws IOException {
		int state = 0;
		if(model.exists()) {
			String cmd = "python3 " + model.getAbsolutePath() + " " +
							device + " " +
							"false " + "true " +
							"[" + brightness + "," + time + "]";
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
				Log.write(Methods.createPrefix() + "Error in AI(264): " + e.getMessage(), false);
			}
		}
		
		return state;
	}
	
	public static void train(String device) {
		new Thread(() -> {
			try {
				File data = new File("AI//data//" + device + ".csv");
				if(data.exists()) {
					String[] cmd = {"python3", "-u" , "AI/Home-System.py",
									data.getName().replace(".csv", ""),
									"true",
									"false"};
					Runtime.getRuntime().exec(cmd);
				}
			}catch(IOException e) {
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in AI(284): " + e.getMessage(), false);
			}
		}).start();
	}
	
	public static void trainNow() {
		Log.write(Methods.createPrefix() + "Starting AI training", true);
		for(String room : Rooms.getRooms()) {
			for(String device : Rooms.getRoomDevices(room)) {
				if(Rooms.getDeviceAIData(room, device)) {
					train(room + "-" + device);
				}
			}
		}
	}
	
}
